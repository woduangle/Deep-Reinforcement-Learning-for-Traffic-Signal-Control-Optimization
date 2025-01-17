# # 创建一个字典
# my_dict = {"apple": ['J1', 'tp_0', 81], "banana": ['J1', 'tp_0', 11], "orange": ['J1', 'tp_0', 22]}
#
# # 遍历字典
# v = [i[2] for _, i in my_dict.items()]
# print(v)
#
a = [1, 2, 3, 4, 5]

for i in a:
    if i < 3:
        print(i)
        continue
    print('i')
