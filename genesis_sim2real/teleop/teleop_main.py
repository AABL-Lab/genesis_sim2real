import genesis_sim2real
from genesis_sim2real.envs.dev_genesis_gym import GenesisGym, _unnormalize_action
import gymnasium as gym
import pygame
import time
import cv2
import numpy as np
def init_gamepad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise Exception("No gamepad found")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick

def test_gamepad(joystick):
    pygame.event.pump()
    # get the joystick state
    axes = joystick.get_axis(0)
    buttons = joystick.get_button(0)
    print(axes, buttons)
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
            print(event.axis, event.value)
        elif event.type == pygame.JOYBUTTONDOWN:
            print(event.button)

def get_action(joystick):
    pygame.event.pump()
    triggers = joystick.get_axis(2), joystick.get_axis(5)
    axes = [-joystick.get_axis(1), -joystick.get_axis(0), -joystick.get_axis(4)]

    # deadzone
    if abs(axes[0]) < 0.075: axes[0] = 0
    if abs(axes[1]) < 0.075: axes[1] = 0
    if abs(axes[2]) < 0.075: axes[2] = 0

    if triggers[0] < 0:
        euler = [0, 0, 0]
        pos = axes
    else:
        euler = axes
        pos = [0, 0, 0]

    buttons = joystick.get_button(0)
    return np.array([*pos, *euler, 1 if buttons else -1])

def main():
    task_name = "genesis_lift-v0"

    joystick = init_gamepad()

    env = GenesisGym()

    done = False; n_episodes = 0
    while n_episodes < 10:
        n_episodes += 1
        obs = env.reset()
        done = False
        print(obs)
        while not done:
            action = get_action(joystick)

            action = _unnormalize_action(action, env.action_space)
            # print(action)
            obs, reward, done, info = env.step(action)
            # for _ in range(10):
            #     env.scene.step()

            img = obs['image']

            # resize image
            img = cv2.resize(img, (0, 0), fx=3, fy=3)

            cv2.imshow('teleop', img)
            cv2.waitKey(1)

            if env.n_steps > 500:
                done = True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_gamepad', action='store_true')
    args = parser.parse_args()

    if args.test_gamepad:
        t0 = time.time()
        # test gamepad
        joystick = init_gamepad()
        while time.time() - t0 < 10:
            print(', '.join(f'{a:.2f}' for a in get_action(joystick)))
            time.sleep(0.05)
    else:
        main()