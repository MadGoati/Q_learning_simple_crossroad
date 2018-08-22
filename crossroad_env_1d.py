import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time

# Define state grid







class environment:

    def __init__(self, ego,car,ped,dim,ego_col,car_row,ped_row):

        self.dim = dim
        self.ego_state = ego
        self.car_state = car
        self.ped_state = ped
        self.width = dim[1]
        self.height = dim[0]
        self.state_value = self.encode()
        self.ego_col = ego_col
        self.car_row = car_row
        self.ped_row = ped_row
        self.ped_done = False
        self.car_done = False
        self.ego_done = False
        self.done = False



    def encode(self):

        if(self.ego_state >= self.height or self.ped_state >= self.width or self.car_state >= self.width):
            print("States don't fit into dimensions")
            return 1

        encode_val = self.ego_state
        encode_val *= self.width
        encode_val += self.car_state
        encode_val *= self.width
        encode_val += self.ped_state

        self.state_value = encode_val
        return encode_val

    def decode(self):

        out = []
        out.append(self.state_value % self.width)
        encode_temp = self.state_value // self.width
        out.append(encode_temp % self.width)
        encode_temp = encode_temp // self.width
        out.append(encode_temp)

        self.ego_state = out[2]
        self.car_state = out[1]
        self.ped_state = out[0]

        assert 0 <= encode_temp < self.height

        return list(reversed(out)) # ego,car,ped

    def render(self):


        state_dim = self.decode()
        ego = state_dim[0]
        car = state_dim[1]
        ped = state_dim[2]


        state_grid = np.zeros(self.dim)
        state_grid[:,self.ego_col] = 4
        state_grid[self.ped_row,:] = 4
        state_grid[self.car_row,:] = 4
        # road: 1 ego: 1 car: 2 ped: 3

        state_grid[self.car_row,car ] =  2
        state_grid[self.ped_row, ped] =  3
        state_grid[ego,self.ego_col] = 1
        # make a color map of fixed colors
        cmap = colors.ListedColormap(
            ['gray', 'green', 'yellow', 'blue','orange'])  # gray:road, green: ego yellow; car blue: pedestrian
        bounds = [0, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.matshow(state_grid, cmap=cmap)

        ax = plt.gca()

        # Major ticks
        ax.set_xticks(np.arange(0, state_grid.shape[1], 1))  # x = colums
        ax.set_yticks(np.arange(0, state_grid.shape[0], 1))  # y = rows

        # Labels for major ticks
        ax.set_xticklabels(np.arange(1, state_grid.shape[1] + 1, 1));
        ax.set_yticklabels(np.arange(1, state_grid.shape[0] + 1, 1));

        # Minor ticks
        ax.set_xticks(np.arange(-.5, state_grid.shape[1], 1), minor=True);
        ax.set_yticks(np.arange(-.5, state_grid.shape[0], 1), minor=True);

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

        # plt.grid()
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        return 0

    def car_action(self):


        if self.car_state + 1 == self.width:
            self.car_done = True

        else:
            self.car_state += 1
            self.car_done = False
            if self.car_state + 1 == self.width:
                self.car_done = True

        return self.car_state, self.car_done

    def ped_action(self):


        if self.ped_state + 1 == self.width:
            self.ped_done = True
        else:

            self.ped_state += 1
            self.ped_done = False

            if self.ped_state + 1 == self.width:
                self.ped_done = True

        return self.ped_state, self.ped_done

    def ego_action(self):

        if self.ego_state + 1 == self.height:
            self.ego_done = True
        else:
            self.ego_state += 1
            self.ego_done = False

        if self.ego_state + 1 == self.height:
            self.ego_done = True

        return self.ego_state, self.ego_done




    def step(self,action):

        self.action = action

        ### Add reward
        self.reward_function()
        #print("Ego:",self.ego_done,"Car:",self.car_done,"Ped:",self.ped_done)

        if(self.ego_done == True and self.car_done and self.ped_done == True ) or self.done == True:
            self.done = True

        else:
            # Move non-ego objects

            self.car_action()
            self.ped_action()
            if action == 1:
                self.ego_action()

            else:
                self.ego_state = self.ego_state
            self.encode() # state_update

            if (self.ego_done == True and self.car_done and self.ped_done == True):
                self.done = True


        return self.state_value,self.reward, self.done

    def reward_function(self):
        self.reward = -1

        #if (self.ego_col == self.ped_state or self.ego_col-1 == self.ped_state) and self.ego_state == self.ped_row-1:
        #    if self.action == 1:
        #        self.reward += -10
        #        self.done = True

        #if (self.ego_col == self.car_state or self.ego_col - 1 == self.car_state) and self.ego_state == self.car_row - 1:
        #    if self.action == 1:
        #        self.reward += -10
        #        self.done = True

        if (self.ego_state == self.height-2):
            if action == 1:
                self.reward = 10
        if (self.ego_state == self.car_row and self.car_state == ego_col):
            self.reward += -10
            self.done = True
        if (self.ego_state == self.ped_row and self.ped_state == ego_col):
            self.reward += -10
            self.done = True


        return self.reward, self.done








# Init States

ego_state = 2

car_state = 1

ped_state = 1

dim = [10,5]

ego_col = 2

car_row = 3

ped_row = 5

done = False

reward_list = []
reward_epsiode = []

learning_rate = 0.9
gamma = 0.9

epsilon = 0.1


env = environment(ego_state,car_state,ped_state,dim,ego_col,car_row,ped_row)
state = env.encode()
env.render()
q_table = np.zeros([dim[0]*dim[1]*dim[1],2])

print(q_table.shape)






for t in range(0, 10000):
    env = environment(ego_state, car_state, ped_state, dim, ego_col, car_row, ped_row)
    state = env.encode()
    #env.render()
    done = False
    while done != True:
        # Decide action
        if np.random.random() < epsilon:
            action = np.random.randint(0,1)
        else:
            action = np.argmax(q_table[state])

        state_1, reward,done = env.step(action)
        reward_list.append(reward)

        # Update Q-table
        q_table[state,action] = (1-learning_rate)*q_table[state,action] + learning_rate*(reward + gamma*max(q_table[state_1]))

        state = state_1
    reward_epsiode.append(sum(reward_list))
    reward_list = []
    if t % 50 == 0:
        print("t:",t,"Average reward:",sum(reward_epsiode)/50)
        reward_epsiode =[]


        #env.render()



print("Training finished")

env = environment(ego_state, car_state, ped_state, dim, ego_col, car_row, ped_row)
state = env.encode()
env.render()
done = False

while done != True:
    action = np.argmax(q_table[state])
    state,reward,done = env.step(action)
    reward_list.append(reward)
    env.render()



reward_total = sum(reward_list)
print("Reward:", reward_total)
print(q_table)

