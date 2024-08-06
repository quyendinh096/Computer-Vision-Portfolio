# Midterm, Individual Porfolio & Capstone

## Midterm
- Object Detection Challenge
- **Objective:** Enhance the performance of an object detection model within specific constraints.

- [Midterm Project Report](https://github.com/quyendinh096/Computer-Vision-Portfolio/blob/1c2dca4469f52d4cb4ed40e08b6396d70998677c/Midterm%2C_Individual_Portfolio_%26_Capstone_Instructions/MT_Quyen%20Dinh_Binte%20Zahra_ITAI_1378-1.docx)
- [Midterm Project Presentation](https://github.com/quyendinh096/Computer-Vision-Portfolio/blob/1c2dca4469f52d4cb4ed40e08b6396d70998677c/Midterm%2C_Individual_Portfolio_%26_Capstone_Instructions/MT_The%20Visionaries_%20QuyenD%20_BinteZahra_JohnMata__ITAI_1378%20-%20PRESENTATION.pptx)
- [Midterm Object Detection Notebook](https://github.com/quyendinh096/Computer-Vision-Portfolio/blob/1c2dca4469f52d4cb4ed40e08b6396d70998677c/Midterm%2C_Individual_Portfolio_%26_Capstone_Instructions/MT_Quyen%20Dinh_Binte%20Zahra_ITAI_1378-1.ipynb)

## Individual Porfolio
- **Objective:** Create a comprehensive portfolio that showcases your learning journey throughout the course. Use a GitHub repository to store and organize your work, and prepare a 10-slide PowerPoint presentation summarizing your portfolio. The deliverable is the presentation with a link to your GitHub repository. 
- [Link to Course Portfolio Github repository](https://github.com/quyendinh096/Computer-Vision-Portfolio.git)
- [Course Portfolio Presentation]()

## Capstone Project: Train an AI Agent to Play Flappy Bird
- **Objective:** To understand and/or implement the process of training an AI agent to play the Flappy Bird game using computer vision and reinforcement learning. This project offers two paths: a conceptual path (no coding required) and a coding path.


Jupyter Notebook Structure
1. Title and Introduction
markdown
Copy code
# Capstone Project: Training an AI Agent to Play Flappy Bird

## Introduction
In this project, we aim to train an AI agent to play the Flappy Bird game using reinforcement learning. We'll use a pre-trained model for feature extraction and implement a Deep Q-Learning (DQN) agent to learn how to play the game.
2. Environment Setup
markdown
Copy code
## Environment Setup

### Installing Libraries
We need to install the required libraries for this project. Execute the following commands to install them:
python
Copy code
!pip install numpy tensorflow keras gym ple opencv-python
markdown
Copy code
### Setting Up Flappy Bird Environment
We will use the PyGame Learning Environment (PLE) to interact with the Flappy Bird game. Let's initialize the environment and set up the preprocessing function.
python
Copy code
import gym
from ple import PLE
from ple.games.flappybird import FlappyBird
import cv2
import numpy as np

# Initialize the game environment
game = FlappyBird()
env = PLE(game, fps=30, display_screen=True)

# Preprocessing function
def preprocess_frame(frame):
    # Convert frame to grayscale and resize
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 84))
    return resized_frame
3. Pre-trained Model Integration
markdown
Copy code
## Pre-trained Model Integration

### Loading and Modifying the Model
We'll use MobileNetV2 for feature extraction. The top layers will be removed, and we'll add custom layers for our Q-network.
python
Copy code
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input

# Load the MobileNetV2 model without the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(84, 84, 3))

# Add custom layers for Q-learning
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(env.action_space.n, activation='linear')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()
4. Reinforcement Learning Implementation
markdown
Copy code
## Reinforcement Learning Implementation

### DQNAgent Class
We'll implement the DQN agent with methods for interacting with the environment and training the model.
python
Copy code
import random
from collections import deque
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
import tensorflow as tf

class DQNAgent:
    def __init__(self, action_space, state_shape):
        self.action_space = action_space
        self.state_shape = state_shape
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Build Q-network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        model = MobileNetV2(weights=None, include_top=False, input_shape=self.state_shape)
        x = model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(self.action_space, activation='linear')(x)
        model = Model(inputs=model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=mean_squared_error)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(np.expand_dims(next_state, axis=0))[0])
            target_f = self.model.predict(np.expand_dims(state, axis=0))
            target_f[0][action] = target
            self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
5. Model Training
markdown
Copy code
## Model Training

### Training Loop
Set up the training loop to interact with the Flappy Bird environment and update the model based on the experiences.
python
Copy code
# Hyperparameters
batch_size = 32
episodes = 1000

# Initialize the DQN agent
agent = DQNAgent(action_space=env.action_space.n, state_shape=(84, 84, 3))

# Training loop
for e in range(episodes):
    env.reset_game()
    state = preprocess_frame(env.getScreenRGB())
    state = np.expand_dims(state, axis=0)
    done = False
    while not done:
        action = agent.act(state)
        reward = env.act(env.getActionSet()[action])
        next_state = preprocess_frame(env.getScreenRGB())
        next_state = np.expand_dims(next_state, axis=0)
        done = env.game_over()
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if e % 10 == 0:
        agent.update_target_model()
    print(f"Episode {e+1}/{episodes} completed.")
6. Testing and Evaluation
markdown
Copy code
## Testing and Evaluation

### Testing the Trained Agent
Evaluate the performance of the trained agent by running it in the environment and recording metrics.
python
Copy code
# Testing the trained agent
test_episodes = 100
total_score = 0

for e in range(test_episodes):
    env.reset_game()
    state = preprocess_frame(env.getScreenRGB())
    state = np.expand_dims(state, axis=0)
    done = False
    while not done:
        action = agent.act(state)
        reward = env.act(env.getActionSet()[action])
        next_state = preprocess_frame(env.getScreenRGB())
        next_state = np.expand_dims(next_state, axis=0)
        done = env.game_over()
        state = next_state
        total_score += reward

average_score = total_score / test_episodes
print(f"Average Score over {test_episodes} episodes: {average_score}")
markdown
Copy code
### Visualization
Plot performance metrics to visualize the agent's learning progress.
python
Copy code
import matplotlib.pyplot as plt

# Example code to plot training performance
# Replace with actual data collection and plotting
episodes = list(range(1, len(rewards) + 1))
plt.plot(episodes, rewards)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Agent Training Performance')
plt.show()
7. Conclusion
markdown
Copy code
## Conclusion

In this project, we trained an AI agent using reinforcement learning to play Flappy Bird. We integrated a pre-trained MobileNetV2 model for feature extraction and implemented a DQN agent for decision-making. The agent was trained and evaluated, and its performance was recorded and analyzed.

### Future Work
Consider exploring other reinforcement learning algorithms, fine-tuning hyperparameters, or improving preprocessing techniques to enhance performance.
