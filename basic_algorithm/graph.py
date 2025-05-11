# -*- coding: utf-8 -*-

from collections import defaultdict, deque

def bfs(graph, start_node):
    # graph: 邻接表, {'node': [sub_node1, sub_node2, ...]}
    # start_node: 广度优先搜索的起始顶点
    # 返回parent和distance
    # 算法完全复现书中22.2伪代码
    
    # 1.初始化状态信息
    color = defaultdict('white')
    parent = defaultdict(None)
    distance = defaultdict(None)
    dq = deque()

    # 2.使用先入先出队列进行广度优先搜索
    color[start_node] = 'gray'
    distance[start_node] = 0
    dq.append(start_node)
    while len(dq) > 0:
        cur_node = dq.popleft()
        for sub_node in graph[cur_node]:
            if color[sub_node] == 'white':
                color[sub_node] = 'gray'
                parent[sub_node] = cur_node
                distance[sub_node] = distance[cur_node] + 1
                dq.append(sub_node)
        # 置黑的操作在前面设置亦可,这里遵循伪代码描述
        color[cur_node] = 'black'

    return parent, distance


def bfs_simple(graph, start_node):
    # 最简化版本,只进行简单遍历
    # 实际上bfs搜索时,不需要三个颜色(状态),保留两个状态即可
    visted = set()
    dq = deque()
    dq.append(start_node)
    visted.add(start_node)
    while len(dq) > 0:
        cur_node = dq.popleft() 
        for sub_node in graph[cur_node]:
            if sub_node not in visted:
                dq.append(sub_node)
                visted.add(sub_node)

def _print_shortest_path(parent, end_node):
    print(end_node)
    if parent[end_node] != None:
        _print_shortest_path(parent, parent[end_node])


def print_shortest_path(graph, start_node, end_node):
    # 借助bfs的搜索结果实现
    parent, _ = bfs(graph, start_node)
    _print_shortest_path(parent, end_node)
    

def print_shortest_path_simple(graph, start_node, end_node):
    # bfs搜索到end_node节点时结束
    # 返回最短路径长度,并打印路径
    if start_node == end_node:
        print(start_node)
        return 0

    visted = set()
    dq = deque()
    parent = defaultdict(None)
    distance = defaultdict(None)

    dq.append(start_node)
    visted.add(start_node)
    distance[start_node] = 0
    
    while len(dq) > 0:
        cur_node = dq.popleft()
        for sub_node in graph[cur_node]:
            if sub_node not in visted:
                parent[sub_node] = cur_node
                distance[sub_node] = distance[cur_node] + 1
                if sub_node == end_node:
                    print(parent, sub_node)
                    return distance[sub_node]




def dfs(graph):
    # 书中伪代码实现
    parent = defaultdict(None)
    color = defaultdict('white')
    d = defaultdict(None)
    f = defaultdict[None]
    t = 0

    def dfs_visit(node):
        t += 1
        d[node] = t
        color[node] = 'gray'
        for sub_node in graph[node]:
            if color[sub_node] == 'white':
                parent[sub_node] = node
                dfs_visit(sub_node) 
            t += 1
            f[node] = t

    for node in graph.keys():
        if color[node] == 'white':
            dfs_visit(node)
    
    return parent, d, f


def dfs_simple(graph):
    # 只进行深度优先遍历,不记录时间
    visted = set()

    def dfs_visit(node):
        visted.add(node)
        for sub_node in graph[node]:
            if sub_node not in visted:
                dfs_visit(sub_node)

    for node in graph.keys():
        if node not in visted:
            dfs_visit(node)


def dfs_with_stack_simple(graph):
    # 只进行深度优先遍历,不记录节点的开始和结束访问时间
    # 每个node在获取子节点后即可pop,因为不需要记录结束时间
    visted = set()
    stack = []

    def dfs_visit(node):
        stack.append(node)
        visted.add(cur_node)
        while len(stack) > 0:
            cur_node = stack.pop()
            for sub_node in graph[cur_node]:
                if sub_node not in visted:
                    visted.add(sub_node)
                    stack.append(sub_node)

    for node in graph.keys():
        if node not in visted:
            dfs_visit(node)


def dfs_with_stack(graph):
    # 实际上color可以用visted取代
    # 入栈的节点必须等到子节点都弹出后才能弹出,因此每个节点实际上需要入栈两次
    # 节点第一次入栈只是暂存,第二次入栈才是真正的访问
    parent = defaultdict(None)
    color = defaultdict('white')
    d = defaultdict(None)
    f = defaultdict(None)
    t = 0
    stack = list()

    def dfs_visit(node):
        stack.append(node)

        while len(stack) > 0:
            cur_node = stack.pop()
            if color[cur_node] == 'white':
                t += 1 
                d[cur_node] = t
                stack.append(cur_node)
                color[cur_node] = 'gray'
                for sub_node in graph[cur_node]:
                    if color[sub_node] == 'white':
                        parent[sub_node] = cur_node
                        stack.append(sub_node)
            elif color[cur_node] == 'gray':
                t += 1
                f[cur_node] = t
                color[cur_node] = 'black'

    for node in graph:
        if color[node] == 'white':
            dfs_visit(node)


def topological_sort(graph):
    # dfs计算f,按照节点的f降序排序
    _, _, f = dfs(graph)
    return sorted(f.items(), key=lambda x: x[1], reverse=True)


def strongly_connected_components(graph):
    # 这版方法只获得了强连通分量,未构建强连通分量间的关系
    # 判断两个强连通分量是否连通很简单,一个强连通分量任意一个节点有到达另一个强连通分量即可可达

    # 深度优先搜索,按照finish time升序排列nodes
    def dfs(graph):
        # 这里只需要获取finish time,可以简化代码
        visited = set()
        # 越靠后finish time越大
        stack = []
        
        def _dfs(graph, node):
            visited.add(node)
            cur_stack = []
            for sub_node in graph[node]:
                if sub_node not in visited:
                    cur_stack.extend(_dfs(graph, sub_node))
            cur_stack.append(node)
            return cur_stack

        for node in graph.keys():
            if node not in visited:
                stack.extend(_dfs(graph, node))
        
        return stack

    sorted_nodes = dfs(graph)
    sorted_nodes = sorted_nodes[::-1]

    # 计算g^T
    graph_t = defaultdict(list)
    for node, sub_nodes in graph.items():
        for sub_node in sub_nodes:
            graph_t[sub_node].append(node)

    # 获取强连通分支
    def build_sccs(graph_t, sorted_nodes):
        visited = set()
        sccs = []

        def _build_scc(graph_t, node):
            visited.add(node)
            scc = []
            for sub_node in graph_t[node]:
                if sub_node not in visited:
                    scc.extend(_build_scc(graph_t, sub_node))
            scc.append(node)
            return scc

        for node in sorted_nodes:
            if node not in visited:
                sccs.extend(_build_scc(graph_t, node))
        
        return sccs

    return build_sccs(graph_t, sorted_nodes)


def check_semiconnected(graph):
    # 半连通有两种情况:图只有一个强连通分支;图有多个强连通分支,但分量图为哈密尔顿链(一条直线)
    # 1.计算强连通分支
    sccs = strongly_connected_components(graph)
    if len(sccs) == 1:
        # 只有一个分量,为半连通
        return True

    # 2.计算分量图
    def build_condensation_graph(graph, sccs):
        c_graph = defaultdict(set)
        node_condensation_map = dict()
        for i in range(len(sccs)):
            for node in sccs[i]:
                node_condensation_map[node] = i
        for i in range(len(sccs)):
            for node in sccs[i]:
                for sub_node in graph[node]:
                    c_graph[i].add(node_condensation_map[sub_node])
        return c_graph

    c_graph = build_condensation_graph(graph, sccs)

    # 3.对分量图进行拓扑排序
    def topological_sort(graph):
        visited = set()
        # 按f升序排序
        stack = []

        def _topological_sort(graph, node):
            visited.add(node)
            cur_stack = []
            for sub_node in graph[node]:
                if sub_node not in visited:
                    cur_stack.extend(_topological_sort(graph, sub_node)) 
            cur_stack.append(node)
            return cur_stack

        for node in graph.keys():
            if node not in visited:
                stack.extend(_topological_sort(graph, node))

    stack = topological_sort(graph)

    # 4.检查分量图是否为哈密尔顿链,即相邻节点是否相连
    # 根据拓扑序
    for i in range(len(stack)-1):
        # 拓扑序大
        u = stack[i+1]
        # 拓扑序小
        v = stack[i]
        
        if c_graph[v] not in c_graph[u]:
            return False
    
    return True


def is_connected_undirected_bfs(graph):
    # 检查是否连通的bfs实现
    if not graph:
        return False

    visited = set()
    # 任选一个作为起点
    start_node = graph.keys()[0]
    dq = deque()
    dq.append(start_node)
    while len(dq) > 0:
        node = dq.popleft()
        visited.add(node)
        for sub_node in graph[node]:
            if sub_node not in visited:
                dq.append(sub_node)
    
    return len(visited) == len(graph)


def is_connected_undirected_dfs(graph):
    # 检查是否连通的dfs实现
    if not graph:
        return False

    visited = set()

    def dfs(graph, node):
        visited.add(node)
        for sub_node in graph[node]:
            if sub_node not in visited:
                dfs(graph, sub_node)

    start_node = graph.keys()[0]
    dfs(graph, start_node)

    return len(visited) == len(graph)


def is_connected_undirected_dfs(graph):
    # 检查是否连通的dfs的非递归实现
    if not graph:
        return False

    visited = set()
    start_node = graph.keys()[0]
    stack = [start_node]
    while len(stack) > 0:
        node = stack.pop()
        visited.add(node)
        for sub_node in graph[node]:
            if sub_node not in visited:
                stack.append(sub_node)
    
    return len(visited) == len(graph)


def is_weakly_connected_directed(graph):
    # 检查有向图是否弱连接
    # 将有向图转为无向图
    undirected_graph = defaultdict(set)
    for node, sub_nodes in graph.items():
        for sub_node in sub_nodes:
            undirected_graph[node].add(sub_node)
            undirected_graph[sub_node].add(node)
    
    return is_connected_undirected_bfs(undirected_graph)


def has_eulerian_circuit_undirected(graph):
    # 无向图欧拉回路判断: 连通&节点度为偶数
    # 充要条件的证明来自deepseek
    # 必要性: 存在欧拉回路的图一定是连通的.由于是回路,入边和出边数量相等,因此每个节点的度为偶数
    # 充分性: 从任一顶点u出发,必可回到u,搜索形成了环C.对剩余未访问的边,由于连通性,必存在C上的顶点w存在未访问的边
    #        将w仿照u进行搜索得到新的路径,和之前的路径合并即得到欧拉回路
    if not is_connected_undirected_bfs(graph):
        return False
    
    for node in graph.keys():
        if len(graph[node]) % 2 != 0:
            return False
    
    return True


def has_eulerian_circuit_directed(graph):
    # 有向图欧拉回路的判断: 弱连通(化简的无向图连通)&节点入度和出度相等
    # 证明方法同无向图
    if not is_weakly_connected_directed(graph):
        return False

    indegree = defaultdict(0)
    outdegree = defaultdict(0)
    for node in graph.keys():
        outdegree[node] = len(graph[node])
        for sub_node in graph[node]:
            indegree[sub_node] += 1
    
    for node in graph.keys():
        if indegree[node] != outdegree[node]:
            return False
    
    return True


def find_euler_circuit_undirected(graph):
    # 从任意一个节点出发进行dfs搜索,为了避免重复,搜索过的路径要删除掉
    # 当一条路径搜索到无路可走时,该路径搜索结束
    # 为了节省空间和避免递归过深,可以直接改graph和用迭代法实现dfs
    # 用迭代法实现dfs操作方便
    if not graph:
        return None

    if not has_eulerian_circuit_undirected(graph):
        return None
    
    # 复制图
    tmp_graph = defaultdict(list)
    for node, sub_nodes in graph.items():
        tmp_graph[node] = list(sub_nodes)

    # 迭代法dfs用到的栈
    stack = []
    # 保存欧拉回路路径
    path = []
    # 任选路径
    node = graph.keys()[0]
    stack.append(node)
    while len(stack) > 0:
        cur_node = stack[-1]
        if tmp_graph[cur_node]:
            sub_node = tmp_graph[cur_node].pop()
            tmp_graph[sub_node].remove(cur_node)
            stack.append(sub_node)
        else:
            path.append(stack.pop())

    return path[::-1]


def find_euler_circuit_directed(graph):
    # 实现方法和无向图相同,区别在于判断是否存在欧拉回路条件略有不同&删除单向边
    if not graph:
        return None

    if not has_eulerian_circuit_directed(graph):
        return None

    # 复制图
    tmp_graph = defaultdict(list)
    for node, sub_nodes in graph.items():
        tmp_graph[node] = list(sub_nodes)

    # 迭代法dfs用到的栈
    stack = []
    # 保存欧拉回路路径
    path = []
    # 任选路径
    node = graph.keys()[0]
    stack.append(node)
    while len(stack) > 0:
        cur_node = stack[-1]
        if tmp_graph[cur_node]:
            # 注意有向图只需要删除单向边
            sub_node = tmp_graph[cur_node].pop()
            stack.append(sub_node)
        else:
            path.append(stack.pop())

    return path[::-1]


def has_euler_path_directed(graph):
    # 无向图欧拉路径的判断条件:
    # - 为欧拉回路
    # - 或者:连通&恰好两个顶点度为奇数,这两个顶点分别为起点和终点
    if not is_connected_undirected_bfs(graph):
        return False
    
    degree = defaultdict(0)
    odd_degree_cnt = 0
    for node in graph.keys():
        degree[node] = len(graph[node])
        if degree[node] % 2 != 0:
            odd_degree_cnt += 1
        
    return (odd_degree_cnt == 0) or (odd_degree_cnt == 2)


def has_euler_path_undirected(graph):
    # 有向图欧拉路径的判断条件:
    # - 为欧拉回路
    # - 或者:弱连通&一个顶点入度=出度+1,一个顶点出度=入度+1,其余顶点入度=出度,这俩特殊顶点一个是终点一个是起点
    if not is_weakly_connected_directed(graph):
        return False

    indegree = defaultdict(0)
    outdegree = defaultdict(0)
    for node in graph.keys():
        outdegree[node] = len(graph[node])
        for sub_node in graph[node]:
            indegree[sub_node] += 1
    
    start_node_cnt = 0
    end_node_cnt = 0
    for node in graph.keys():
        if indegree[node] == outdegree[node]:
            continue
        elif indegree[node] == outdegree[node]+1:
            end_node_cnt += 1
        elif outdegree[node] == indegree[node]+1:
            start_node_cnt += 1
        else:
            return False

    return ((start_node_cnt == 0) and (end_node_cnt == 0)) or ((start_node_cnt == 1) and (end_node_cnt == 1))


def find_euler_path_undirected(graph):
    # 获取欧拉路径时,必须从奇数度的点开始,因为偶数度的度进入和离开次数相等.如果从偶数度的节点开始搜索搜索得到的路径是断开的
    # 比如a-b-c-d-e,如果从b开始搜,那么dfs节点入栈顺序是e-d-c-a-b,并不是一条完整的路径
    if not graph:
        return None

    if not has_euler_path_undirected(graph):
        return None

    # 找到搜索起点
    start_node = graph.keys()[0]
    for node in graph.keys():
        if len(graph[node]) % 2 != 0:
            start_node = node
            break

    # 搜索路径,Hierholzer算法,希尔霍尔泽算法
    tmp_graph = defaultdict(set)
    for node, sub_nodes in graph.items():
        tmp_graph[node] = set(sub_nodes)

    path = []
    stack = [start_node]
    while len(start_node) > 0:
        cur_node = stack[-1]
        if tmp_graph[cur_node]:
            sub_node = tmp_graph[cur_node].pop()
            tmp_graph[sub_node].remove(cur_node)
            stack.append(sub_node)
        else:
            path.append(stack.pop())

    return path[::-1]


def find_euler_path_directed(graph):
    # 获取欧拉路径时,从出度=入度+1的节点开始搜
    if not graph:
        return None

    if not has_euler_path_directed(graph):
        return None

    # 找到搜索起点
    start_node = graph.keys()[0]
    indegree = defaultdict(0)
    outdegree = defaultdict(0)
    for node in graph.keys():
        outdegree[node] = len(graph[node])
        for sub_node in graph[node]:
            indegree[sub_node] += 1
    for node in graph.keys():
        # 入度比出度小1
        if indegree[node] == outdegree[node]-1:
            start_node = node
            break

    # 搜索路径,Hierholzer算法
    tmp_graph = defaultdict(set)
    for node, sub_nodes in graph.items():
        tmp_graph[node] = set(sub_nodes)
    
    path = []
    stack = [start_node]
    while len(stack) > 0:
        cur_node = stack[-1]
        if tmp_graph[cur_node]:
            sub_node = tmp_graph[cur_node].pop()
            stack.append(sub_node)
        else:
            path.append(stack.pop())

    return path[::-1]