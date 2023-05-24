# Relevant imports
from manim import *
import numpy as np
import pandas as pd

# Activation functions
def relu(X):
    return np.maximum(0,X)

# def softmax(X):
#     return np.exp(X)/sum(np.exp(X))

# stable softmax
def softmax(X):
    Z = X - max(X)
    numerator = np.exp(Z)
    denominator = np.sum(numerator)
    return numerator/denominator

# Calculates the output of a given layer
def calculate_layer_output(w, prev_layer_output, b, activation_type="relu"):
    # Steps 1 & 2
    g = w @ prev_layer_output + b

    # Step 3
    if activation_type == "relu":
        return relu(g)
    if activation_type == "softmax":
        return softmax(g)

# Initialize weights & biases
def init_layer_params(row, col):
    w = np.random.randn(row, col)
    b = np.random.randn(row, 1)
    return w, b

# Calculate ReLU derivative
def relu_derivative(g):
    derivative = g.copy()
    derivative[derivative <= 0] = 0
    derivative[derivative > 0] = 1
    return np.diag(derivative.T[0])

def layer_backprop(previous_derivative, layer_output, previous_layer_output
                   , w, activation_type="relu"):
    # 1. Calculate the derivative of the activation func
    dh_dg = None
    if activation_type == "relu":
        dh_dg = relu_derivative(layer_output)
    elif activation_type == "softmax":
        dh_dg = softmax_derivative(layer_output)

    # 2. Apply chain rule to get derivative of Loss function with respect to:
    dL_dg = dh_dg @ previous_derivative # activation function

    # 3. Calculate the derivative of the linear function with respect to:
    dg_dw = previous_layer_output.T     # a) weight matrix
    dg_dh = w.T                         # b) previous layer output
    dg_db = 1.0                         # c) bias vector

    # 4. Apply chain rule to get derivative of Loss function with respect to:
    dL_dw = dL_dg @ dg_dw               # a) weight matrix
    dL_dh = dg_dh @ dL_dg               # b) previous layer output
    dL_db = dL_dg * dg_db               # c) bias vector

    return dL_dw, dL_dh, dL_db

def gradient_descent(w, b, dL_dw, dL_db, learning_rate):
    w -= learning_rate * dL_dw
    b -= learning_rate * dL_db
    return w, b

def get_prediction(o):
    return np.argmax(o)

# Compute Accuracy (%) across all training data
def compute_accuracy(train, label, w1, b1, w2, b2, w3, b3):
    # Set params
    correct = 0
    total = train.shape[0]

    # Iterate through training data
    for index in range(0, total):
        # Select a single data point (image)
        X = train[index: index+1,:].T

        # Forward pass: compute Output/Prediction (o)
        h1 = calculate_layer_output(w1, X, b1, activation_type="relu")
        h2 = calculate_layer_output(w2, h1, b2, activation_type="relu")
        o = calculate_layer_output(w3, h2, b3, activation_type="softmax")

        # If prediction matches label Increment correct count
        if label[index] == get_prediction(o):
            correct+=1

    # Return Accuracy (%)
    return (correct / total) * 100


# Calculate Softmax derivative
def softmax_derivative(o):
    derivative = np.diag(o.T[0])

    for i in range(len(derivative)):
        for j in range(len(derivative)):
            if i == j:
                derivative[i][j] = o[i] * (1 - o[i])
            else:
                derivative[i][j] = -o[i] * o[j]
    return derivative


class VisualiseNeuralNetwork(Scene):

    # Global Variables
    ANIMATION_RUN_TIME = 0.2
    HEADER_FONT_SIZE = 20
    HEADER_2_FONT_SIZE = 15
    HEADER_3_FONT_SIZE = 11
    HEADER_HEIGHT = -3.6
    HEATMAP_SQUARE_SCALE = 0.07

    TRAINING_DATA_POINTS = [1, 0, 24, 13, 32, 8, 21, 6, 10, 11]
    DIGIT_X_PLACEMENTS = [5.5, 4.75, 4, 3.25, 2.5, 5.5, 4.75, 4, 3.25, 2.5]
    DIGIT_Y_PLACEMENTS = [-2, -2, -2, -2, -2, 2.5, 2.5, 2.5, 2.5, 2.5]

    W2_PLACEMENT = [6.25, 0]
    H1_PLACEMENT = [4.75, 0]
    W2H1_PLACEMENT = [3.1, 0]
    B2_PLACEMENT = [2.1, 0]
    H2_2_PLACEMENT = [1, 0]

    W3_PLACEMENT = [-1, 0]
    H2_PLACEMENT = [-2.5, 0]
    W3H2_PLACEMENT = [-4.15, 0]
    B3_PLACEMENT = [-5.25, 0]
    O_PLACEMENT = [-6.35, 0]
    PREDICTIONS_X_PLACEMENT = [-2, -3, -4, -5, -6, -2, -3, -4, -5, -6]
    PREDICTIONS_Y_PLACEMENT = [-2, -2, -2, -2, -2, 2.5, 2.5, 2.5, 2.5, 2.5]

    INPUT_IMAGES = []
    HEATMAP_W2      = None
    HEATMAP_H1      = None
    HEATMAP_B2      = None
    HEATMAP_W2H1    = None
    HEATMAP_H2      = None
    HEATMAP_H2_2    = None
    HEATMAP_W3      = None
    HEATMAP_W3H2    = None
    HEATMAP_O       = None
    HEATMAP_B3      = None
    PREDICTIONS_OBJECTS = []

    OUTPUT_H1 = np.zeros((10, 10))
    OUTPUT_W2H1 = np.zeros((10, 10))
    OUTPUT_H2 = np.zeros((10, 10))
    OUTPUT_W3H2 = np.zeros((10, 10))
    OUTPUT_O = np.zeros((10, 10))
    OUTPUT_PREDICTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def construct(self):
        ### INITIALISE NEURAL NET PARAMETERS ###
        # Extract MNIST csv data into train & test variables
        train = np.array(pd.read_csv('train.csv', delimiter=','))
        test = np.array(pd.read_csv('test.csv', delimiter=','))

        # Extract the first column of the training dataset into a label array
        label = train[:, 0]
        # The train dataset now becomes all columns except the first
        train = train[:, 1:]

        # Initialise vector of all zeroes with 10 columns and the same number
        # of rows as the label array
        Y = np.zeros((label.shape[0], 10))

        # assign a value of 1 to each column index matching the label value
        Y[np.arange(0, label.shape[0]), label] = 1.0

        # Normalize test & training dataset
        train = train / 255
        test = test / 255

        # Set hyperparameter(s)
        learning_rate = 0.01

        # Set other params
        epoch = 0
        previous_accuracy = 100
        accuracy = 0

        # Randomly initialize weights & biases
        w1, b1 = init_layer_params(10, 784)  # Hidden Layer 1
        w2, b2 = init_layer_params(10, 10)  # Hidden Layer 2
        w3, b3 = init_layer_params(10, 10)  # Output Layer

        ### CREATE SCENE ###
        h2_to_h2_arrow = VGroup()
        input_image_text = None

        for i in range(len(self.TRAINING_DATA_POINTS)):
            # Create input image
            training_image = train[
                                self.TRAINING_DATA_POINTS[i]:self.TRAINING_DATA_POINTS[i] + 1
                                , :
                             ].T
            self.INPUT_IMAGES += [self.create_input_image(training_image, self.DIGIT_X_PLACEMENTS[i], self.DIGIT_Y_PLACEMENTS[i])]
            # Create header for input image & add to scene
            input_image_text = self.create_text("Input", self.HEADER_2_FONT_SIZE, 0 , 0)
            input_image_text.next_to(self.INPUT_IMAGES[i], 0.5 * UP)

            self.play(Create(self.INPUT_IMAGES[i])
                      , Write(input_image_text)
                      , run_time=self.ANIMATION_RUN_TIME)

            # Create prediction text & add to scene
            self.PREDICTIONS_OBJECTS += [self.create_prediction_text("Prediction", "...", self.PREDICTIONS_X_PLACEMENT[i], self.PREDICTIONS_Y_PLACEMENT[i])]
            self.add(self.PREDICTIONS_OBJECTS[i])

        # Create heatmaps
        # Hidden Layer 2
        self.HEATMAP_W2 = self.create_heatmap(self.W2_PLACEMENT[0], self.W2_PLACEMENT[1], w2, self.HEATMAP_SQUARE_SCALE, "w2")
        self.HEATMAP_B2 = self.create_heatmap(self.B2_PLACEMENT[0], self.B2_PLACEMENT[1], b2, self.HEATMAP_SQUARE_SCALE, "b2")
        self.HEATMAP_H1     = self.create_heatmap(self.H1_PLACEMENT[0], self.H1_PLACEMENT[1], self.OUTPUT_H1, self.HEATMAP_SQUARE_SCALE, "h1")
        self.HEATMAP_W2H1   = self.create_heatmap(self.W2H1_PLACEMENT[0], self.W2H1_PLACEMENT[1], self.OUTPUT_W2H1, self.HEATMAP_SQUARE_SCALE, "w2@h1")
        self.HEATMAP_H2_2   = self.create_heatmap(self.H2_2_PLACEMENT[0], self.H2_2_PLACEMENT[1], self.OUTPUT_H2, self.HEATMAP_SQUARE_SCALE, "h2")

        h2_2_symbol_text = self.create_text("=", self.HEADER_2_FONT_SIZE, 0, 0)
        h2_2_symbol_text.next_to(self.HEATMAP_H2_2, 0.25 * LEFT)
        w2h1_symbol_text = self.create_text("=", self.HEADER_2_FONT_SIZE, 0, 0)
        w2h1_symbol_text.next_to(self.HEATMAP_W2H1, 0.25 * LEFT)
        b2_symbol_text = self.create_text("+", self.HEADER_2_FONT_SIZE, 0, 0)
        b2_symbol_text.next_to(self.HEATMAP_B2, 0.3 * LEFT)

        # Output Layer
        self.HEATMAP_H2     = self.create_heatmap(self.H2_PLACEMENT[0], self.H2_PLACEMENT[1], self.OUTPUT_H2, self.HEATMAP_SQUARE_SCALE, "h2")
        self.HEATMAP_W3H2   = self.create_heatmap(self.W3H2_PLACEMENT[0], self.W3H2_PLACEMENT[1], self.OUTPUT_W3H2, self.HEATMAP_SQUARE_SCALE, "w3@h2")
        self.HEATMAP_O      = self.create_heatmap(self.O_PLACEMENT[0], self.O_PLACEMENT[1], self.OUTPUT_O, self.HEATMAP_SQUARE_SCALE, "o")
        self.HEATMAP_W3     = self.create_heatmap(self.W3_PLACEMENT[0], self.W3_PLACEMENT[1], w3, self.HEATMAP_SQUARE_SCALE, "w3")
        self.HEATMAP_B3     = self.create_heatmap(self.B3_PLACEMENT[0], self.B3_PLACEMENT[1], b3, self.HEATMAP_SQUARE_SCALE, "b3")

        w3h2_symbol_text = self.create_text("=", self.HEADER_2_FONT_SIZE, 0, 0)
        w3h2_symbol_text.next_to(self.HEATMAP_W3H2, 0.5 * LEFT)
        b3_symbol_text = self.create_text("+", self.HEADER_2_FONT_SIZE, 0, 0)
        b3_symbol_text.next_to(self.HEATMAP_B3, 0.5 * LEFT)
        o_symbol_text = self.create_text("=", self.HEADER_2_FONT_SIZE, 0, 0)
        o_symbol_text.next_to(self.HEATMAP_O, 0.5 * LEFT)

        for heatmap in [self.HEATMAP_H1, self.HEATMAP_W2H1, self.HEATMAP_H2_2, self.HEATMAP_H2, self.HEATMAP_W3H2, self.HEATMAP_O]:
            heatmap_header = self.create_text("0 1 2 3 4 5 6 7 8 9", self.HEADER_3_FONT_SIZE, 0, 0)
            heatmap_header.next_to(heatmap, 0.25 * DOWN)
            heatmap_header.scale(0.95)
            self.add(heatmap_header)

        # Create arrows & braces
        line3 = Line(start=self.HEATMAP_H2_2.get_bottom() + (0.4 * DOWN), end=self.HEATMAP_H2_2.get_bottom() + (0.7 * DOWN), stroke_width=0.9)
        line4 = Line(start=line3, end=np.array([self.HEATMAP_H2.get_top()[0], line3.get_end()[1], 0]), stroke_width=0.9)
        arrow_2 = Arrow(start=DOWN/4, end=ORIGIN, stroke_width=0.9)
        arrow_2.move_to(line4.get_end())
        arrow_2.shift((arrow_2.get_length()/2) * UP)
        h2_to_h2_arrow.add(line3, line4, arrow_2)

        # Animate creation of nodes & connections
        self.play(
            Create(self.HEATMAP_H2)
            , Create(self.HEATMAP_W3H2)
            , Create(self.HEATMAP_O)
            , Create(self.HEATMAP_H1)
            , Create(self.HEATMAP_W2H1)
            , Create(self.HEATMAP_H2_2)
            , Create(h2_to_h2_arrow)
            , Create(self.HEATMAP_W2)
            , Create(self.HEATMAP_B2)
            , Create(self.HEATMAP_W3)
            , Create(self.HEATMAP_B3)
            , Write(b2_symbol_text)
            , Write(w3h2_symbol_text)
            , Write(b3_symbol_text)
            , Write(o_symbol_text)
            , Write(w2h1_symbol_text)
            , Write(h2_2_symbol_text)
            , Write(input_image_text)
            , run_time=self.ANIMATION_RUN_TIME
        )

        # Create headers to distinguish the different layers & add to scene
        hidden_layer2_text = self.create_text("Hidden Layer 2", self.HEADER_FONT_SIZE, 4, self.HEADER_HEIGHT)
        output_text = self.create_text("Output Layer", self.HEADER_FONT_SIZE, -3.5, self.HEADER_HEIGHT)
        self.add(hidden_layer2_text)
        self.add(output_text)

        # Create status text & add to scene
        status_text = self.create_text(f'Epoch: {0}\nAccuracy: {0:.2f}%'
                                       , self.HEADER_FONT_SIZE
                                       , -6.15
                                       , -3.65)
        self.add(status_text)

        ## NEURAL NET TRAINING ###
        # While:
        #  1. Accuracy is improving by 1% or more per epoch, and
        #  2. There are 20 epochs or less
        while (accuracy < 80 or abs(accuracy - previous_accuracy) >= 1) and epoch <= 20:
            print(f'------------- Epoch {epoch} -------------')

            # record previous accuracy
            previous_accuracy = accuracy

            # Iterate through training data
            for index in range(train.shape[0]):
                # Select a single image and associated y vector
                X = train[index:index + 1, :].T
                y = Y[index:index + 1].T

                # 1. Forward pass: compute Output/Prediction (o)
                h1 = calculate_layer_output(w1, X, b1, activation_type="relu")
                h2 = calculate_layer_output(w2, h1, b2, activation_type="relu")
                o = calculate_layer_output(w3, h2, b3, activation_type="softmax")

                # 2. Compute Loss Vector
                L = np.square(o - y)

                # 3. Backpropagation
                # Compute Loss derivative w.r.t. Output/Prediction vector (o)
                dL_do = 2.0 * (o - y)

                # Compute Output Layer derivatives
                dL3_dw3, dL3_dh2, dL3_db3 = layer_backprop(dL_do, o, h2, w3
                                                           , "softmax")
                # Compute Hidden Layer 2 derivatives
                dL2_dw2, dL2_dh2, dL2_db2 = layer_backprop(dL3_dh2, h2, h1, w2
                                                           , "relu")
                # Compute Hidden Layer 1 derivatives
                dL1_dw1, _, dL1_db1 = layer_backprop(dL2_dh2, h1, X, w1
                                                     , "relu")

                # 4. Update weights & biases
                w1, b1 = gradient_descent(w1, b1, dL1_dw1, dL1_db1, learning_rate)
                w2, b2 = gradient_descent(w2, b2, dL2_dw2, dL2_db2, learning_rate)
                w3, b3 = gradient_descent(w3, b3, dL3_dw3, dL3_db3, learning_rate)

                # Decide whether to animate
                animate = True if index in self.TRAINING_DATA_POINTS else False

                # Set animation parameters
                if animate:
                    # print(h1[:, 0].shape)
                    # print(self.OUTPUT_H1[:, i].shape)

                    i = self.TRAINING_DATA_POINTS.index(index)
                    self.OUTPUT_PREDICTIONS[i] = get_prediction(o)
                    self.OUTPUT_H1[:, i] = h1[:, 0]
                    self.OUTPUT_W2H1[:, i] = (w2 @ h1)[:, 0]
                    self.OUTPUT_H2[:, i] = h2[:, 0]
                    self.OUTPUT_W3H2[:, i] = (w3 @ h2)[:, 0]
                    self.OUTPUT_O[:, i] = o[:, 0]

            # Animate Changes
            # Hidden Layer 2
            self.animate_heatmap(self.HEATMAP_W2, self.W2_PLACEMENT[0], self.W2_PLACEMENT[1], w2,
                                                  self.HEATMAP_SQUARE_SCALE, "w2")
            self.animate_heatmap(self.HEATMAP_B2, self.B2_PLACEMENT[0], self.B2_PLACEMENT[1], b2,
                                                  self.HEATMAP_SQUARE_SCALE, "b2")
            self.animate_heatmap(self.HEATMAP_H1, self.H1_PLACEMENT[0], self.H1_PLACEMENT[1],
                                                  self.OUTPUT_H1, self.HEATMAP_SQUARE_SCALE, "h1")
            self.animate_heatmap(self.HEATMAP_W2H1, self.W2H1_PLACEMENT[0], self.W2H1_PLACEMENT[1],
                                                    self.OUTPUT_W2H1, self.HEATMAP_SQUARE_SCALE, "w2@h1")
            self.animate_heatmap(self.HEATMAP_H2_2, self.H2_2_PLACEMENT[0], self.H2_2_PLACEMENT[1],
                                                    self.OUTPUT_H2, self.HEATMAP_SQUARE_SCALE, "h2")

            # Output Layer
            self.animate_heatmap(self.HEATMAP_H2, self.H2_PLACEMENT[0], self.H2_PLACEMENT[1],
                                                  self.OUTPUT_H2, self.HEATMAP_SQUARE_SCALE, "h2")
            self.animate_heatmap(self.HEATMAP_W3H2, self.W3H2_PLACEMENT[0], self.W3H2_PLACEMENT[1],
                                                    self.OUTPUT_W3H2, self.HEATMAP_SQUARE_SCALE, "w3@h2")
            self.animate_heatmap(self.HEATMAP_O, self.O_PLACEMENT[0], self.O_PLACEMENT[1], self.OUTPUT_O,
                                                 self.HEATMAP_SQUARE_SCALE, "o")
            self.animate_heatmap(self.HEATMAP_W3, self.W3_PLACEMENT[0], self.W3_PLACEMENT[1], w3,
                                                  self.HEATMAP_SQUARE_SCALE, "w3")
            self.animate_heatmap(self.HEATMAP_B3, self.B3_PLACEMENT[0], self.B3_PLACEMENT[1], b3,
                                                  self.HEATMAP_SQUARE_SCALE, "b3")

            for i in range(len(self.OUTPUT_PREDICTIONS)):
                self.animate_prediction_text("Prediction", self.PREDICTIONS_OBJECTS[i], self.OUTPUT_PREDICTIONS[i],
                                             self.PREDICTIONS_X_PLACEMENT[i], self.PREDICTIONS_Y_PLACEMENT[i])

            # Compute & print Accuracy (%)
            accuracy = compute_accuracy(train, label, w1, b1, w2, b2, w3, b3)
            print(f'Accuracy: {accuracy:.2f} %')

            # Increment epoch
            epoch += 1
            self.animate_text(status_text, f'Epoch: {epoch}\nAccuracy: {accuracy:.2f}%', self.HEADER_FONT_SIZE, -6.15, -3.65)

        self.wait(3)

    # Create Methods
    def create_input_image(self, training_image, left_shift, down_shift):
        # Initialise params
        square_count = training_image.shape[0]
        rows = np.sqrt(square_count)

        # Create list of squares to represent pixels
        squares = [
            Square(fill_color=WHITE
                   , fill_opacity=training_image[i]
                   , stroke_width=0.2).scale(0.01)
            for i in range(square_count)
        ]

        # Place all the squares into a VGroup and arrange into a 28x28 grid
        group = VGroup(*squares).arrange_in_grid(rows=int(rows), buff=0)

        # Shift into correct position in the scene
        group.shift(left_shift * LEFT).shift(down_shift * DOWN)

        return group

    def create_text(self, text, font_size, left_shift, down_shift):
        # Create text
        text = Text(text, font_size=font_size)

        # Position text
        text.shift(left_shift * LEFT)
        text.shift(down_shift * DOWN)

        return text

    def create_prediction_text(self, header_text, prediction, left_shift, down_shift):
        # Create group
        prediction_text_group = VGroup()

        # Create & position text
        prediction_text = Text(f'{prediction}', font_size=40)
        prediction_text.shift(left_shift * LEFT).shift(down_shift * DOWN)

        # Create text box (helps with positioning Prediction Header)
        prediction_text_box = Square(fill_opacity=0
                                     , stroke_opacity=0
                                     , side_length=0.75)
        prediction_text_box.move_to(prediction_text)

        # Create Header Text
        prediction_header = Text(header_text
                                 , font_size=self.HEADER_2_FONT_SIZE)
        prediction_header.next_to(prediction_text_box, UP)

        # Group items
        prediction_text_group.add(prediction_header)
        prediction_text_group.add(prediction_text)
        prediction_text_group.add(prediction_text_box)

        return prediction_text_group

    def create_heatmap(self, left_shift, down_shift, array, scale, text, text_shift=UP):
        # Initialise params
        rows, cols = array.shape
        square_count = rows * cols
        norm_array = array.copy() if np.max(array) == 0 else array.copy() / np.max(np.absolute(array))
        group = VGroup()
        text = self.create_text(text
                                , self.HEADER_2_FONT_SIZE
                                , 0
                                , 0)

        # Create list of squares to represent pixels
        squares = [
            Square(fill_color=GREEN if norm_array.flatten()[i] >= 0 else RED
                   , fill_opacity=abs(norm_array.flatten()[i])
                   , stroke_width=0.3).scale(scale)
            for i in range(square_count)
        ]

        # Place all the squares into a VGroup and arrange into a 28x28 grid
        for square in squares:
            group.add(square)
        group.arrange_in_grid(rows=int(rows), buff=0)

        # Shift into correct position in the scene
        group.shift(left_shift * LEFT).shift(down_shift * DOWN)

        # Add text
        text.next_to(group, text_shift)
        group.add(text)

        return group

    # Animate Methods
    def animate_input_image(self, input_image, X, left_shift, down_shift):
        # 1. Create input image with new parameters
        new_input_image = self.create_input_image(X, left_shift, down_shift)
        # 2. Transform old input image to new image
        self.play(Transform(input_image, new_input_image)
                  , run_time=self.ANIMATION_RUN_TIME)

    def animate_text(self, text, new_string, font_size, left_shift, down_shift):
        # 1. Create text with new parameters
        new_text = self.create_text(new_string
                                    , font_size
                                    , left_shift
                                    , down_shift)
        # 2. Transform old text to new text
        self.play(Transform(text, new_text)
                  , run_time=self.ANIMATION_RUN_TIME)

    def animate_prediction_text(self, header_text, prediction_text_group, prediction, left_shift, down_shift):
        # 1. Create prediction text with new parameters
        new_prediction_text_group = self.create_prediction_text(header_text, prediction
                                                                , left_shift, down_shift)
        # 2. Transform old prediction text to new prediction text
        self.play(Transform(prediction_text_group, new_prediction_text_group)
                  , run_time=self.ANIMATION_RUN_TIME)
        # self.play(Circumscribe(prediction_text, Circle))

    def animate_heatmap(self, heatmap, left_shift, down_shift, array, scale, text, text_shift=UP):
        # 1. Create heatmap with new parameters
        new_heatmap = self.create_heatmap(left_shift, down_shift, array, scale, text, text_shift)
        # 2. Transform old heatmap to new heatmap
        self.play(Transform(heatmap, new_heatmap)
                  , run_time=self.ANIMATION_RUN_TIME)
