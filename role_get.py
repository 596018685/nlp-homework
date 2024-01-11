def role_data(data_load="/root/data/rool.txt"):
    with open(data_load, 'r') as f:
        role_data=[]
        for line in f.readlines():
            role_data.append(line)
    return role_data

def weapon_data(data_load="/root/data/weapon.txt"):
    with open(data_load, 'r') as f:
        weapon_data=[]
        for line in f.readlines():
            weapon_data.append(line)
    return weapon_data