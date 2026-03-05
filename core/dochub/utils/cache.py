from redis import ConnectionPool, Redis, Sentinel, RedisCluster
from redis.cluster import ClusterNode

from config import CONFIG


def create_redis_connection(redis_conf: dict):
    mode = redis_conf.get("mode", "single")
    hosts = redis_conf.get("hosts", ["localhost:6379"])
    password = redis_conf.get("password", None)
    max_connections = redis_conf.get("max_connections")
    if mode == 'single':
        ip, port = hosts[0].split(':')
        pool = ConnectionPool(host=ip, port=int(port), db=0, max_connections=max_connections,
                              password=password, decode_responses=True)
        return Redis(connection_pool=pool)
    elif mode == 'cluster':
        startup_nodes = [ClusterNode(*node.split(':')) for node in hosts]
        return RedisCluster(
            startup_nodes=startup_nodes,
            skip_full_coverage_check=True, decode_responses=True, password=password
        )
    elif mode == 'sentinel':
        my_sentinel = Sentinel(
            [tuple(node.split(':')) for node in hosts],
            sentinel_kwargs={"password": password},
            decode_responses=True
        )
        return my_sentinel.master_for("mymaster", db=0, password=password)
        # 暂未实现读写分离
        # slave_conn = my_sentinel.slave_for("mymaster", db=0, password=password)
    else:
        raise ValueError(f"Unsupported service mode: {mode}")


redis_client = create_redis_connection(CONFIG["redis"])
