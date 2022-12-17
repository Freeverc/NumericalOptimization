import matplotlib.pyplot as plt


x = []
cost = []
lr = []
log_name = "data/abalone_scale.log0.txt"
# log_name = "data/bodyfat_scale.log0.txt"
# log_name = "data/housing_scale.log2.txt"
with open(log_name, "r") as f:
    line = f.readline()
    while line: 
        data = line.split()
        # print(data)
        line = f.readline()
        x.append(int(data[0]))
        cost.append(float(data[1]))
        lr.append(float(data[2]))
        if len(x)>80:
            break

if log_name[-5] == '0':
    plt.title('GRADIENT_DECENT')
elif log_name[-5] == '1':
    plt.title('CONJUGATE_GRADIENT')
elif log_name[-5] == '2':
    plt.title('QUASI_NEWTON')
plt.xlabel('iteration')
plt.ylabel('cost ')
plt.plot(x, cost)
# plt.show()
plt.savefig(log_name+'.png')