#Last resort to see implementation from class
import numpy as np
from matplotlib import pyplot as plt, lines


IN = open('perceptron_training_data.tsv')
#OUT = open('NNout.')
text = IN.readlines()

X = []
bgn = True
y = []

for line in text:
    if bgn:
        bgn = False
        continue

    row = line.strip().split('\t')
    #print(row)
    X.append([float(row[0]),float(row[1])])

    y.append([int(row[2])])

w,b = zip(X,y)

def step_function(output):
    if output > 0:
        return 1
    return 0

epochs = 25
bias = np.random.rand(1)
weights = np.random.rand(2)
learning_rate = 0.1

#----------------------------------------
# need to find decision boundaries, we set them initially and then get value from function
decision_boundaries = []
for i in X:
    xdata = i[0]
    ydata = i[1]

xdata = [i[0] for i in w]
ydata = [i[1] for i in w]

maxx = max(xdata)
minx = min(xdata)
maxy = max(ydata)
miny = min(ydata)

def get_decision_boundary():
    global decision_boundaries
    y0 = (-bias - minx*weights[0]) / weights[1]
    y1 = (-bias - maxx*weights[1]) / weights[1]
    decision_boundaries.append([y0[0], y1[0]])
    
def make_fig():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim([minx-0.5, maxx+0.5])
    ax.set_ylim([miny-0.5, maxy+0.5])
    ax.scatter(xdata, ydata, c=b)
    ax.plot([minx, maxx], decision_boundaries[-1])
    ax.set_title(f'epoch {len(decision_boundaries)}')
    plt.show()

# New we begin to train yh is pred, 
get_decision_boundary()
make_fig()
for i in range(epochs):
    correct = 0
    for xx, yy in zip(w, b):
        y_hat = step_function(bias+np.dot(xx, weights))
        if y_hat != yy and y_hat == 0:
            weights = weights + learning_rate*xx
            bias = bias + learning_rate
        elif y_hat != yy and y_hat == 1: # pred is wrong, adjust
            weights = weights - learning_rate*xx
            bias = bias - learning_rate
        elif y_hat == yy: # pred is correct and add to count
            correct += 1
    get_decision_boundary()
    make_fig()
    accuracy = correct/len(y)*100
    print('=========EPOCH========= {}'.format(i+1))
    print('Accuracy: {}%'.format(accuracy))
    print('Weights:', weights)
    print('Bias:', bias)
    db = decision_boundaries[-1]
    print('decision boundary', (minx,db[0]), (maxx, db[1]))
    
    if correct == len(b):
        print('Converged after {} epochs.'.format(i+1))
        break


#need to circle back on figure from matplotlib

