# Neural Network Algorithm

The project uses Stochastic Gradient Descent and Sigmoid during forward pass & back propagation phases. 
Couldn't have gotten this far without concepts & explanations from 'Make Your Own Neural Network by Tariq Rashid' (Part one).

The neural network is written using only Go's standard library without external dependencies. 

## Live Demo
The canvas on the grid gets translated into a 28 x 28 uint8 array consisting of grayscale values from 0 to 255. This array gets converted to JSON then sent to an local Gin API endpoint. Once the network guesses (around 23ms) the network puts its guess into a 0 - 9 index array.

### Images
Blank
<img width="1879" height="947" alt="image" src="https://github.com/user-attachments/assets/37adaeaa-d03e-4370-8179-b1e2a4ea1381" />

Seven
<img width="1879" height="947" alt="image" src="https://github.com/user-attachments/assets/2e5d8230-9c84-4764-8750-63f106afd055" />

Two
<img width="1879" height="947" alt="image" src="https://github.com/user-attachments/assets/3bfc8b07-55f1-44ad-bc26-7b5126e86b60" />

Zero
<img width="1879" height="947" alt="image" src="https://github.com/user-attachments/assets/da3257ed-650e-408f-9ea2-77c40797cf96" />

Three
<img width="1879" height="947" alt="image" src="https://github.com/user-attachments/assets/2d37d586-6aa9-4141-a184-83501729efe2" />



## Future Tasks
Due to MNIST Training bias, recognition works best when the number is in the center for the grid. 
