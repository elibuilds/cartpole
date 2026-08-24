import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(episodes, is_training=False, render=False):

    #initialize the frozenlake environment
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None) 

    #conditional check to either train the agent or use trained q table
    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # if train we initialize a new q table
    else:
        f = open('frozen_lake8x8.pkl', 'rb')    #else we load our trained q table
        q = pickle.load(f)
        f.close()

    # hyperparameters we can adjust

    learning_rate = 1.0
    discount_factor_g = 0.9

    epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = 0.99997
    final_epsilon = 0.01         
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes) #keep track of number of rewards obtained in episode although max reward possible is 1.

    # loop to move through the environment depending on the number of episodes
    for i in range(episodes):
        
        state = env.reset()[0]     # reset the environment at the beginning of each episode 
        terminated = False 
        truncated = False

        while(not terminated and not truncated ):
            # select action to make based on epsilon greedy formula and also whether we are training 
            if is_training and rng.random() < epsilon:  
                action = env.action_space.sample()
            else: 
                action = np.argmax(q[state, :])

            new_state,reward,terminated,truncated,_ = env.step(action)     # obtain new env variables after taking action 

            # if training.. update q table wwith new q table with new state-action value obtained 
            if is_training:
                q[state, action] = q[state, action] + learning_rate*(reward + discount_factor_g*np.max(q[new_state, :])- q[state, action])

            # set current state to the new state obtained
            state = new_state

        # decrease epsilon 
        epsilon = max(final_epsilon, epsilon*epsilon_decay)

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    print(q) # print trained q table 

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.savefig("frozen_lake8x8.png")

    if is_training:
        f = open("frozen_lake8x8.pkl", "wb")
        pickle.dump(q,f)
        f.close()


if __name__ == '__main__':
    run(100000, is_training=True) #adjust number of episodes 

    # comment out above run command and uncomment below to view agent use trained q table
    #run(1,render='human', is_training=False) 