import torch
import numpy as np
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet

def get_state(game):
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

def main():
    model_path = 'model/model.pth'
    
    model = Linear_QNet(11, 256, 3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    game = SnakeGameAI()
    
    while True:
        state_old = get_state(game)
        
        state_tensor = torch.tensor(state_old, dtype=torch.float)
        prediction = model(state_tensor)
        action = torch.argmax(prediction).item()
        
        final_move = [0, 0, 0]
        final_move[action] = 1
        _, done, score = game.play_step(final_move)
        
        if done:
            print(f"Game Over! Score: {score}")
            game.reset()

if __name__ == "__main__":
    main()