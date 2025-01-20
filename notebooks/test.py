import gymnasium as gym
from time import sleep
env= gym.make('CartPole-v1',render_mode='human')
(state,_)=env.reset()

episodeNumber=5
timeSteps=100
 
 
for episodeIndex in range(episodeNumber):
    initial_state=env.reset()
    print(episodeIndex)
    env.render()
    new_env = env.reached


    if(new_env==)

    appendedObservations=[]
    for timeIndex in range(timeSteps):
        print(timeIndex)
        random_action=env.action_space.sample()
        observation, reward, terminated, truncated, info =env.step(random_action)
        appendedObservations.append(observation)
        sleep(0.1)
        if (terminated):
            break
env.close()   