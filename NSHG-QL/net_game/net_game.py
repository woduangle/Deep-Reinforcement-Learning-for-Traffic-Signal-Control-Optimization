import numpy as np
import pandas as pd
from net_game import matrix
from net_game import lemkeHowson

player_list = {}  # 所有leader对应的flower{j3: [j1, j6], j13:[j11, j15]}
ne_payoff_table = {}  # nash中leader收益矩阵 DataFrame全存放
nash_q_values = {}  # nash中的Q值
se_payoff_table = {}  # Stackelberg中收益矩阵
se_q_values = {}  # Stackelberg中的Q值


def init_player_list(agent_setting):
    """初始化收益矩阵"""
    global player_list
    global ne_payoff_table
    global se_payoff_table
    for key, val in agent_setting.items():
        if agent_setting[key]['identity'] == 'Leader':
            player_list[key] = agent_setting[key]["asymmetric_opponent"]  # 初始化得到player list
            leader_opponent = agent_setting[key]["leader_opponent"]  # 初始化得到每个leader的博弈对手
            # 初始化收益矩阵，行列标签为leader的动作及其对手动作
            ne_payoff_table[key] = pd.DataFrame(0, index=agent_setting[key]['actions']['names'],
                                                columns=agent_setting[leader_opponent]['actions']['names'])
            se_payoff_table[key] = {}  # 初始化 SE 收益矩阵字典，其中包含每个 leader 对应的多个 follower 收益矩阵
            for opponent in player_list[key]:
                # 初始化每个 leader 对应的 follower 收益矩阵
                se_payoff_table[key][opponent] = pd.DataFrame(0, index=agent_setting[key]['actions']['names'],
                                                              columns=agent_setting[opponent]['actions']['names'])
                # 初始化独立的 follower 收益矩阵
                se_payoff_table[opponent] = pd.DataFrame(0, index=agent_setting[key]['actions']['names'],
                                                         columns=agent_setting[opponent]['actions']['names'])
        else:
            pass

def get_player_list():
    global player_list
    return player_list

def get_leaders_NE_values_by():
    """得到收益矩阵及nash的q值"""
    global player_list
    global nash_q_values
    leaders = list(player_list.keys())  # leader的agent列：['J3', 'J13']
    m0, m1 = construct_payoff_table(leaders)  # m0,m1为两个leader的收益矩阵
    q_values = calculate_nash_ql_values(m0=m0, m1=m1)  # 根据收益矩阵，计算得到nash的q值
    nash_q_values = dict(zip(leaders, q_values))  # zip将列表转成字典，例：{‘J3’：0.0， ‘J13’：0.0}


def construct_payoff_table(leaders):
    """计算收益矩阵"""
    global ne_payoff_table
    player0_payoff_table = ne_payoff_table[leaders[0]]  # J3的收益矩阵
    player1_payoff_table = ne_payoff_table[leaders[1]]  # J13的收益矩阵
    # index 表示 DataFrame 的行索引，即收益矩阵的行标签；tolist() 将行索引转换为一个包含索引值(即所有动作名称)的列表
    player0_action_names = player0_payoff_table.index.tolist()
    player1_action_names = player1_payoff_table.index.tolist()
    # Matrix 类的构造函数，该构造函数接受两个参数，即行数和列数，用于创建一个矩阵对象，分别构建两个leader的收益矩阵
    m0 = matrix.Matrix(player0_payoff_table.shape[0], player0_payoff_table.shape[1])  # shape[0] 就是行数；shape[1] 就是列数
    m1 = matrix.Matrix(player1_payoff_table.shape[1], player1_payoff_table.shape[0])
    for i in range(player0_payoff_table.shape[0]):
        for j in range(player1_payoff_table.shape[1]):
            # 将收益表中对应行动的收益赋值给leader的矩阵
            m0.setItem(i + 1, j + 1, player0_payoff_table[player0_action_names[i]][player1_action_names[j]])
            m1.setItem(i + 1, j + 1, player1_payoff_table[player1_action_names[i]][player0_action_names[j]])
    return m0, m1  # 返回两个leader的收益矩阵


def calculate_nash_ql_values(m0, m1):
    """计算nash的q值"""
    (m0, m1) = (m0, m1)  # m0,m1为两个leader的收益矩阵
    # 使用LemkeHowson算法，用于找到二个leader的零和博弈的纳什均衡，保存至包含两个元素的的元组，分别表示两个leader的策略概率
    probprob = lemkeHowson.lemkeHowson(m1=m0, m2=m1)
    prob0 = np.array(probprob[0])  # 将策略概率转换为 NumPy 数组
    prob1 = np.array(probprob[1])
    # 将 NumPy 数组转换为 NumPy 矩阵，其中 prob1 被重新形状为列向量
    prob0 = np.matrix(prob0)
    prob1 = np.matrix(prob1).reshape((-1, 1))
    # calculate the nash values
    m_m0 = []
    m_m1 = []
    # 双重循环通过矩阵的行和列，将矩阵中的元素逐个添加到相应的列表中
    for i in range(m0.getNumRows()):
        for j in range(m0.getNumCols()):
            m_m0.append(m0.getItem(i + 1, j + 1))
    for i in range(m1.getNumRows()):
        for j in range(m1.getNumCols()):
            m_m1.append(m1.getItem(i + 1, j + 1))
    # 将列表重新转换为矩阵，并恢复其原始形状
    m_m0 = np.matrix(m_m0).reshape((m0.getNumRows(), m0.getNumCols()))
    m_m1 = np.matrix(m_m1).reshape((m1.getNumRows(), m1.getNumCols()))
    # 计算 Nash 均衡的 Q 值
    m_nash0 = prob0 * m_m0 * prob1
    m_nash1 = prob0 * m_m1 * prob1
    # 通过索引 [0, 0] 获取了该矩阵的第一行第一列的元素；.nom()用于获取分数的分子部分、denom()用于获取分数的分母部分
    nash0 = m_nash0[0, 0].nom() / m_nash0[0, 0].denom()  # 将分数的分子除以分母，得到一个浮点数或有理数的表达式
    nash1 = m_nash1[0, 0].nom() / m_nash1[0, 0].denom()
    nash_q_values = [nash0, nash1]  # 将计算得到的 Nash 均衡值放入一个列表，并返回该列表
    return nash_q_values


def update_leader_NE_payoff_table(player1, player2, player1_action, player2_action, player1_q, player2_q):
    """更新收益矩阵"""
    global player_list
    global ne_payoff_table
    # 如果leader当前的收益矩阵值小于新的 Q 值，则更新收益矩阵中对应位置的值；另一个对手leader同理
    if ne_payoff_table[player1].loc[player1_action, player2_action] < player1_q:
        ne_payoff_table[player1].loc[player1_action, player2_action] = player1_q
    if ne_payoff_table[player2].loc[player2_action, player1_action] < player2_q:
        ne_payoff_table[player2].loc[player2_action, player1_action] = player2_q



"""-----Stackelberg------"""


def get_agents_SE_value_by(agent_setting, tls_agents):
    """计算每个 agent 的 Stackelberg equilibrium (SE) 收益值"""
    global se_q_values
    global se_payoff_table
    for key, val in agent_setting.items():
        if agent_setting[key]['identity'] == 'Leader':
            se_value_leader = {}
            # 遍历当前 leader 的每个 follower 以及对应的收益矩阵
            for follower, follower_se_payoff_table in se_payoff_table[key].items():
                # 获取当前 leader 和 follower 的收益矩阵
                leader_payoff_table = se_payoff_table[key][follower]
                follower_payoff_table = se_payoff_table[follower]
                # 获取当前 leader 执行的动作
                leader_action = tls_agents[key].action_current
                # 初始化变量，用于存储 follower 收益矩阵中最大值及其对应位置
                argmax_follower_value = 0
                argmax_follower_location = []
                for follower_action in agent_setting[follower]['actions']['names']:
                    # 检查当前 follower 动作对应的收益是否大于之前记录的最大值
                    if follower_payoff_table.loc[leader_action, follower_action] >= argmax_follower_value:
                        # 更新最大值和对应位置
                        argmax_follower_value = follower_payoff_table.loc[leader_action, follower_action]
                        argmax_follower_location = [leader_action, follower_action]
                    else:
                        pass
                # 根据 leader中所有follower 最优动作的位置，获取 leader 的收益值
                argmax_leader_value = leader_payoff_table.loc[argmax_follower_location[0], argmax_follower_location[1]]
                # 将 follower 的 Stackelberg equilibrium 收益值添加到字典和全局变量中。
                se_value_leader[follower] = argmax_leader_value
                se_q_values[follower] = argmax_follower_value
            # 将 leader 对应的所有 follower 的 Stackelberg equilibrium 收益值添加到全局变量中
            se_q_values[key] = se_value_leader
        else:
            pass


def update_agents_SE_payoff_table(agent_setting, tls_agents):
    """更新 Stackelberg equilibrium (SE) 收益矩阵"""
    global se_q_values
    global se_payoff_table
    for key, val in agent_setting.items():
        if agent_setting[key]['identity'] == 'Leader':
            for opponent in agent_setting[key]['asymmetric_opponent']:
                """SE_agent_list = [key, opponent]"""
                # 获取当前 leader 和对手的状态、动作、Q表以及在其Q表中的SE收益值
                state_list = [tls_agents[key].state_current, tls_agents[opponent].state_current]
                action_list = [tls_agents[key].action_current, tls_agents[opponent].action_current]
                q_table_list = [tls_agents[key].q_table, tls_agents[opponent].q_table]
                se_values_list = [q_table_list[0].loc[state_list[0], action_list[0]],
                                  q_table_list[1].loc[state_list[1], action_list[1]]]
                # 如果当前 leader 和对手的 SE 收益值大于其在收益矩阵中的值，则更新收益矩阵
                if se_payoff_table[key][opponent].loc[action_list[0], action_list[1]] < se_values_list[0]:
                    se_payoff_table[key][opponent].loc[action_list[0], action_list[1]] = se_values_list[0]
                if se_payoff_table[opponent].loc[action_list[0], action_list[1]] < se_values_list[1]:
                    se_payoff_table[opponent].loc[action_list[0], action_list[1]] = se_values_list[1]
        else:
            pass
