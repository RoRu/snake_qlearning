from pygame.locals import *
from PIL import Image
from agent import DQAgent
from logger import Logger
import numpy as np
import pygame
import random
import sys

MAX_EPISODE_LENGTH_FACTOR = 100
MAX_EPISODES_BETWEEN_TRAININGS = 1500
STEP = 20
APPLE_SIZE = 20
SCREEN_SIZE = 300
START_X = SCREEN_SIZE / 2 - 5 * STEP
START_Y = SCREEN_SIZE / 2 - STEP
BACKGROUND_COLOR = (0, 0, 0)
SNAKE_COLOR = (255, 255, 255)
APPLE_COLOR = (255, 255, 255)
ACTIONS = 4
SCREENSHOT_DIMS = (84, 84)
APPLE_REWARD = 1
DEATH_REWARD = -1
LIFE_REWARD = 0


def reset_game():
    global xs, ys, dirs, score, episode_length, episode_reward, applepos, s, \
        action, state, next_state, must_die
    xs = [START_Y,
          START_Y,
          START_Y,
          START_Y,
          START_Y]
    ys = [START_X + 5 * STEP,
          START_X + 4 * STEP,
          START_X + 3 * STEP,
          START_X + 2 * STEP,
          START_X]
    dirs = random.choice([0, 1, 3])
    score = 0
    episode_length = 0
    episode_reward = 0
    must_die = False
    applepos = (random.randint(0, SCREEN_SIZE - APPLE_SIZE),
                random.randint(0, SCREEN_SIZE - APPLE_SIZE))

    action = random.randint(0, ACTIONS - 1)

    state = [screenshot(), screenshot()]
    next_state = [screenshot(), screenshot()]

    s.fill(BACKGROUND_COLOR)
    for j in range(0, len(xs)):
        s.blit(img, (xs[j], ys[j]))
    s.blit(appleimage, applepos)
    pygame.display.update()


def collide(x1, x2, y1, y2, w1, w2, h1, h2):
    return x1 + w1 > x2 and x1 < x2 + w2 and y1 + h1 > y2 and y1 < y2 + h2


def die():
    global logger, remaining_iters, score, episode_length, episode_reward, \
        must_test, experience_buffer, exp_backup_counter, global_episode_counter

    global_episode_counter += 1

    if global_episode_counter > MAX_EPISODES_BETWEEN_TRAININGS:
        logger.log("Something's gone wrong")
        DQA.quit()
        sys.exit(0)

    if must_test:
        logger.to_csv('test_data.csv', [score, episode_length, episode_reward])
        logger.log('Test episode - Score: %s; Steps: %s'
                   % (score, episode_length))

    must_test = False

    if score >= 1 and episode_length >= 10:
        exp_backup_counter += len(experience_buffer)
        print 'Score: %s; Episode length: %s' \
              % (score, episode_length)
        logger.to_csv('train_data.csv', [score, episode_length, episode_reward])
        print '%s samples of %s' % (exp_backup_counter, DQA.batch_size)
        for exp in experience_buffer:
            DQA.add_experience(*exp)

    if DQA.must_train():
        exp_backup_counter = 0
        logger.log('Episodes elapsed: %d' % global_episode_counter)
        global_episode_counter = 0

        if remaining_iters == 0:
            DQA.quit()
            sys.exit(0)

        DQA.train()

        remaining_iters -= 1 if remaining_iters != -1 else 0

        must_test = True
        logger.log('Test episode')

    experience_buffer = []

    pygame.display.update()
    reset_game()


def screenshot():
    global s
    data = pygame.image.tostring(s, 'RGB')
    image = Image.frombytes('RGB', (SCREEN_SIZE, SCREEN_SIZE), data)
    image = image.convert('L')
    image = image.resize(SCREENSHOT_DIMS)

    image = image.convert('1')
    matrix = np.asarray(image.getdata(), dtype=np.float64)
    return matrix.reshape(image.size[0], image.size[1])


remaining_iters = -1

logger = Logger()
logger.log({
    'Action space': ACTIONS,
    'Reward apple': APPLE_REWARD,
    'Reward death': DEATH_REWARD,
    'Reward life': LIFE_REWARD
})
logger.to_csv('test_data.csv', ['score,episode_length,episode_reward'])
logger.to_csv('train_data.csv', ['ecore,episode_length,episode_reward'])
logger.to_csv('loss_history.csv', ['loss'])

DQA = DQAgent(ACTIONS, logger=logger)
experience_buffer = []

score = 0
episode_length = 0
episode_reward = 0
episode_nb = 0
exp_backup_counter = 0
global_episode_counter = 0
must_test = False

xs = [START_Y,
      START_Y,
      START_Y,
      START_Y,
      START_Y]
ys = [START_X + 5 * STEP,
      START_X + 4 * STEP,
      START_X + 3 * STEP,
      START_X + 2 * STEP,
      START_X]
dirs = random.choice([0, 1, 3])
must_die = False
applepos = (random.randint(0, SCREEN_SIZE - APPLE_SIZE),
            random.randint(0, SCREEN_SIZE - APPLE_SIZE))

pygame.init()
s = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption('Snake')
appleimage = pygame.Surface((APPLE_SIZE, APPLE_SIZE))
appleimage.fill(APPLE_COLOR)
img = pygame.Surface((STEP, STEP))
img.fill(SNAKE_COLOR)
clock = pygame.time.Clock()

action = random.randint(0, ACTIONS - 1)

state = [screenshot(), screenshot()]
next_state = [screenshot(), screenshot()]

while True:
    episode_length += 1
    reward = LIFE_REWARD
    next_state[0] = state[1]

    clock.tick()
    for e in pygame.event.get():
        if e.type == QUIT:
            DQA.quit()
            sys.exit(0)

    if action == 2 and dirs != 0:
        dirs = 2
    elif action == 0 and dirs != 2:
        dirs = 0
    elif action == 3 and dirs != 1:
        dirs = 3
    elif action == 1 and dirs != 3:
        dirs = 1

    if collide(xs[0], applepos[0],
               ys[0], applepos[1],
               STEP, APPLE_SIZE,
               STEP, APPLE_SIZE):
        score += 1
        reward = APPLE_REWARD
        xs.append(700)
        ys.append(700)
        applepos = (random.randint(0, SCREEN_SIZE - APPLE_SIZE),
                    random.randint(0, SCREEN_SIZE - APPLE_SIZE))

    i = len(xs) - 1
    while i >= 2:
        if collide(xs[0], xs[i],
                   ys[0], ys[i],
                   STEP, STEP,
                   STEP, STEP):
            must_die = True
            reward = DEATH_REWARD
        i -= 1

    if xs[0] < 0 or xs[0] > SCREEN_SIZE - APPLE_SIZE * 2 or ys[0] < 0 or ys[0] > SCREEN_SIZE - APPLE_SIZE * 2:
        must_die = True
        reward = DEATH_REWARD

    i = len(xs) - 1
    while i >= 1:
        xs[i] = xs[i - 1]
        ys[i] = ys[i - 1]
        i -= 1
    if dirs == 0:
        ys[0] += STEP
    elif dirs == 1:
        xs[0] += STEP
    elif dirs == 2:
        ys[0] -= STEP
    elif dirs == 3:
        xs[0] -= STEP

    s.fill(BACKGROUND_COLOR)
    for i in range(0, len(xs)):
        s.blit(img, (xs[i], ys[i]))
    s.blit(appleimage, applepos)
    pygame.display.update()

    next_state[1] = screenshot()

    experience_buffer.append((np.asarray([state]), action, reward,
                              np.asarray([next_state]),
                              True if must_die else False))
    episode_reward += reward

    state = list(next_state)

    action = DQA.get_action(np.asarray([state]), testing=must_test)

    if must_die or episode_length > len(xs) * MAX_EPISODE_LENGTH_FACTOR:
        die()
