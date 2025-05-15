# agent_checkpoint.py
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os


class Agent:
    def __init__(self):
        self.lr = 0.001
        self.gamma = 0.9
        self.epsilon = 100
        self.max_memory = 100000
        self.memory = deque(maxlen=self.max_memory)
        self.batch_size = 1000
        self.n_games = 0
        self.record = 0
        
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)
        
        self.checkpoint_path = os.path.join("model", "checkpoint.pth")
        self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.n_games = checkpoint["n_games"]
                self.epsilon = checkpoint["epsilon"]
                self.record = checkpoint["record"]
                print(f"Load Checkpoint: STEP {self.n_games}, epsilon={self.epsilon}")
            except Exception as e:
                print(f"Load Checkpoint Failed: {e}")
                self.reset_checkpoint()
        else:
            self.reset_checkpoint()

    def reset_checkpoint(self):
        self.save_checkpoint(force=True)

    def save_checkpoint(self, force=False):
        if force or self.n_games % 10 == 0:
            os.makedirs("model", exist_ok=True)
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                "n_games": self.n_games,
                "epsilon": self.epsilon,
                "record": self.record
            }
            torch.save(checkpoint, self.checkpoint_path)
            print(f"Save Checkpoint: STEP {self.n_games}")

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_d)) or 
            (dir_l and game.is_collision(point_u)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_d)) or 
            (dir_r and game.is_collision(point_u)),
            
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(2, self.epsilon - 1)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    
    if not os.path.exists('model'):
        os.makedirs('model')
    
    agent = Agent()
    game = SnakeGameAI()
    print(agent.epsilon)
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.remember(state_old, final_move, reward, state_new, done)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > agent.record:
                agent.record = score
                agent.model.save()
                agent.save_checkpoint()
            
            if agent.n_games % 10 == 0:
                agent.save_checkpoint()
            
            print('Game', agent.n_games, 'Score', score, 'Record:', agent.record)
            
            #if agent.n_games % 10 == 0 or score > agent.record:
            #    agent.save_checkpoint()
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()