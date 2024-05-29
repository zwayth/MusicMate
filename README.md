# Music Recommendation System using Reinforcement Learning

![License](https://img.shields.io/badge/license-MIT-blue.svg)

This project implements a music recommendation system using reinforcement learning techniques. The system prompts users to provide their music preferences through a set of interactions and then generates personalized song recommendations based on their input.

## Overview

The recommendation system utilizes a dataset of songs containing various features and genres. Users can register with a name and receive a unique user ID, or they can log in using their existing user ID. The system then prompts users to select song features they like and rate a set number of songs to understand their preferences. Based on this data, the system employs reinforcement learning to generate personalized song recommendations for each user.

## Installation

To run the music recommendation system, follow these steps:

1. Clone this repository to your local machine:

```bash 
git clone https://github.com/yourusername/MusicMate.git
```


2. Navigate to the project directory:

```bash
cd MusicMate
```


3. Install the required Python packages:

```bash
pip install -r requirements.txt
```


4. Ensure you have a dataset of songs in CSV format located in the "../data/" directory relative to your script file. The CSV file should contain information about songs, including features and genres.

## Usage

To use the music recommendation system, follow these steps:

1. Run the Python script `recommendation_system.py`:

```bash
python recommendation_system.py
```


2. If you're a new user, choose to register and enter your name. The system will generate a unique user ID for you. If you're an existing user, choose to log in and enter your user ID.

3. Follow the prompts to select song features you like and rate a set number of songs to provide your preferences.

4. Based on your preferences, the system will generate personalized song recommendations for you.

## Features

- User Registration and Login: Users can register with a name and receive a unique user ID, or they can log in using their existing user ID.
- Personalized Recommendations: The system employs reinforcement learning techniques to generate personalized song recommendations based on user preferences.
- Interactive Interface: Users can interact with the system through a set of prompts to provide their music preferences.

## Dependencies

- pandas==1.3.3
- numpy==1.21.2
- plotly==5.3.1
- tqdm==4.62.3
- scikit-learn==0.24.2
- theano==1.0.5

## Contributing

Contributions to this project are welcome. You can contribute by submitting bug reports, feature requests, or pull requests through the GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



