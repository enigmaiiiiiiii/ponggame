import numpy as np
import pickle
import gym

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # 200×6400,"Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
    图像压缩以及二值化
    """

    I = I[35:195]  # crop 沿第一维度剪裁
    I = I[::2, ::2, 0]  # downsample by factor of 2  ::2的含义，从开始到结束以步长为2选择，将160×160压缩成（80，80），并选择第一层
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()  # rave:平铺，由ravel方法赋值的数组会改变原数组


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward
    pi(s|a) = r+gamma*pi(s1|a1)+gamma^2*pi(s2|a2)+....
    返回计算值，标签值"""
    discounted_r = np.zeros_like(r)  # size相同的全零数组
    running_add = 0
    for t in reversed(range(r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    """返回隐藏层状态，动作概率"""
    h = np.dot(model['W1'], x)  # 全连接200×6000乘6000=DIM(200)
    h[h < 0] = 0  # ReLU nonlinearity  激活函数Relu消除线性关系
    logp = np.dot(model['W2'], h)  # 隐藏层？
    p = sigmoid(logp)  # 动作概率
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)  # 状态处理
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)  #  特征处理，当前帧减去上一帧作为神经网络输入
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)  # 只产生向上移动的概率
    """返回动作概率,隐藏层状态"""
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice! 根据概率动作

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation x: 状态 xs:状态集合
    hs.append(h)  # hidden state  h:hidden layer parameters
    y = 1 if action == 2 else 0  # a "fake label" y代表动作标签0=动作3，1=动作2
    dlogps.append(y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward  # sum of reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        # print('ep {0}: game finished, total reward: {1}'.format(episode_number, reward_sum))
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)  # vstack 沿维度1堆叠
        eph = np.vstack(hs)  # 将list堆叠成ndarray,作为经验池
        epdlogp = np.vstack(dlogps)  # 动作经验池
        epr = np.vstack(drs)  # reward 经验池
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)  # 一个完整过程，每一步的状态和动作的价值，size = 状态size
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)  # inplace操作
        discounted_epr /= np.std(discounted_epr)  # reward标准化

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                """optimizer RMSprop"""
                g = grad_buffer[k]  # gradient 梯度=g
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('episode:{2}\treward total was {0}\trunning mean:{1}'.format(reward_sum, running_reward,episode_number))
        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    # if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
    #     print('ep {0}: game finished, reward: {1}'.format(episode_number, reward))
