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
    ANIMATION_RUN_TIME = 0.1
    HEADER_FONT_SIZE = 20
    HEADER_HEIGHT = -3.5
    TRAINING_DATA_POINT = 1

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
        # Create input image
        training_image = train[
                            self.TRAINING_DATA_POINT:self.TRAINING_DATA_POINT + 1
                            , :
                         ].T
        input_image = self.create_input_image(training_image, left_shift=5)

        # Create the nodes for each of the layers
        num_nodes = 10
        h1_node_group, h1_nodes_list = self.create_nodes(3, 0.25, num_nodes)
        h2_node_group, h2_nodes_list = self.create_nodes(0, 0.25, num_nodes)
        o_node_group, o_nodes_list = self.create_nodes(-3, 0.25, num_nodes)

        # Create connections
        connections_1 = self.create_connections(h1_nodes_list, h2_nodes_list, w2)
        connections_2 = self.create_connections(h2_nodes_list, o_nodes_list, w3)

        # Create headers to distinguish the different layers & add to scene
        hidden_layer1_text = self.create_text("Hidden Layer 1"
                                              , self.HEADER_FONT_SIZE
                                              , 3
                                              , self.HEADER_HEIGHT)
        hidden_layer2_text = self.create_text("Hidden Layer 2"
                                              , self.HEADER_FONT_SIZE
                                              , 0
                                              , self.HEADER_HEIGHT)
        output_text = self.create_text("Output Layer"
                                       , self.HEADER_FONT_SIZE
                                       , -3
                                       , self.HEADER_HEIGHT)

        self.add(hidden_layer1_text)
        self.add(hidden_layer2_text)
        self.add(output_text)

        # Create header for input image & add to scene
        input_image_text = self.create_text("Input Image"
                                            , self.HEADER_FONT_SIZE
                                            , 0
                                            , 0)
        input_image_text.next_to(input_image, UP)
        self.add(input_image_text)

        # Create prediction text & add to scene
        prediction_text_group = self.create_prediction_text("...", -5.5)
        self.add(prediction_text_group)

        # Create status text & add to scene
        status_text = self.create_text(f'Epoch: {0}\nAccuracy: {0:.2f}%'
                                       , self.HEADER_FONT_SIZE
                                       , -5.5
                                       , -3)
        self.add(status_text)

        # Animate creation of nodes & connections
        self.play(Create(input_image)
                  , Create(h1_node_group)
                  , Create(h2_node_group)
                  , Create(connections_1)
                  , Create(o_node_group)
                  , Create(connections_2)
                  )

        ### NEURAL NET TRAINING ###
        # While:
        #  1. Accuracy is improving by 1% or more per epoch, and
        #  2. There are 20 epochs or less
        while (accuracy < 75 or abs(accuracy - previous_accuracy) >= 1) and epoch <= 12:
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
                animate = True if index == self.TRAINING_DATA_POINT else False

                # Animate change
                if animate:
                    self.animate_input_image(input_image, X, 5)
                    self.animate_nodes(h1_node_group, h1, 3, 0.25, num_nodes)
                    self.animate_nodes(h2_node_group, h2, 0, 0.25, num_nodes)
                    self.animate_connections(h1_nodes_list, h2_nodes_list
                                             , connections_1, w2)
                    self.animate_nodes(o_node_group, o, -3, 0.25, num_nodes)
                    self.animate_connections(h2_nodes_list, o_nodes_list
                                             , connections_2, w3)
                    self.animate_prediction_text(prediction_text_group
                                                 , get_prediction(o), -5)

            # Compute & print Accuracy (%)
            accuracy = compute_accuracy(train, label, w1, b1, w2, b2, w3, b3)
            print(f'Accuracy: {accuracy:.2f} %')

            # Increment epoch
            epoch += 1
            self.animate_text(status_text
                            , f'Epoch: {epoch}\nAccuracy: {accuracy:.2f}%'
                            , self.HEADER_FONT_SIZE
                            , -5.5
                            , -3)
        self.wait(2)

    # Create Methods
    def create_input_image(self, training_image, left_shift):
        # Initialise params
        square_count = training_image.shape[0]
        rows = np.sqrt(square_count)

        # Create list of squares to represent pixels
        squares = [
            Square(fill_color=WHITE
                   , fill_opacity=training_image[i]
                   , stroke_width=0.5).scale(0.03)
            for i in range(square_count)
        ]

        # Place all the squares into a VGroup and arrange into a 28x28 grid
        group = VGroup(*squares).arrange_in_grid(rows=int(rows), buff=0)

        # Shift into correct position in the scene
        group.shift(left_shift * LEFT)

        return group

    def create_nodes(self, left_shift, down_shift, num_nodes, layer_output=None):
        # Create VGroup & list to hold created nodes
        node_group = VGroup()
        nodes = []

        # Create list of circles to represent nodes
        for i in range(num_nodes):
            # Set fill opacity to 0
            opacity = 0.0
            text = "0.0"
            # If a layer output has been passed and the max value is not 0
            if layer_output is not None and np.max(layer_output) != 0.0:
                # Set opacity as normalised layer output value
                opacity = (layer_output[i] / np.max(np.absolute(layer_output)))[0]
                # Set text as layer output
                text = f'{layer_output[i][0]:.1f}'

            # Create node
            node = Circle(radius=0.23
                , stroke_color=WHITE
                , stroke_width=0.7
                , fill_color=GRAY
                , fill_opacity=opacity
                )

            # Add to nodes list
            nodes += [node]

            fill_text = Text(text, font_size=12)
            # Position fill text in circle
            fill_text.move_to(node)

            # Group fill text and node and add to node_group
            group = VGroup(node, fill_text)
            node_group.add(group)


        # Arrange & position node_group
        node_group.arrange(DOWN, buff=0.2)
        node_group.shift(left_shift * LEFT).shift(down_shift * DOWN)

        return node_group, nodes

    def create_connections(self, left_layer_nodes, right_layer_nodes, w):
        # Create VGroup to hold created connections
        connection_group = VGroup()

        # Iterate through right layer nodes
        for l in range(len(right_layer_nodes)):
            # Iterate through left layer nodes
            for r in range(len(left_layer_nodes)):
                # Calculate opacity from normalised weight matrix values
                opacity = 0.0 if np.max(np.absolute(w[l, :])) == 0.0 \
                    else w[l, r] / np.max(np.absolute(w[l, :]))
                # Set colour
                colour = GREEN if opacity >= 0 else RED

                # Create connection line
                line = Line(start=right_layer_nodes[l].get_edge_center(LEFT)
                            , end=left_layer_nodes[r].get_edge_center(RIGHT)
                            , color=colour
                            , stroke_opacity=abs(opacity)
                            )

                # Add to connection group
                connection_group.add(line)
        return connection_group

    def create_text(self, text, font_size, left_shift, down_shift):
        # Create text
        text = Text(text, font_size=font_size)

        # Position text
        text.shift(left_shift * LEFT)
        text.shift(down_shift * DOWN)

        return text

    def create_prediction_text(self, prediction, left_shift):
        # Create group
        prediction_text_group = VGroup()

        # Create & position text
        prediction_text = Text(f'{prediction}', font_size=40)
        prediction_text.shift(left_shift * LEFT)

        # Create text box (helps with positioning Prediction Header)
        prediction_text_box = Square(fill_opacity=0
                                     , stroke_opacity=0
                                     , side_length=0.75)
        prediction_text_box.move_to(prediction_text)

        # Create Header Text
        prediction_header = Text("Prediction"
                                 , font_size=self.HEADER_FONT_SIZE)
        prediction_header.next_to(prediction_text_box, UP)

        # Group items
        prediction_text_group.add(prediction_header)
        prediction_text_group.add(prediction_text)
        prediction_text_group.add(prediction_text_box)

        return prediction_text_group

    # Animate Methods
    def animate_input_image(self, input_image, X, left_shift):
        # 1. Create input image with new parameters
        new_input_image = self.create_input_image(X, left_shift)
        # 2. Transform old input image to new image
        self.play(Transform(input_image, new_input_image)
                  , run_time=self.ANIMATION_RUN_TIME)

    def animate_nodes(self, layer_group, layer_output
                      , left_shift, down_shift, num_neurons):
        # 1. Create nodes with new parameters
        new_layer_group, _ = self.create_nodes(left_shift
                                               , down_shift
                                               , num_neurons
                                               , layer_output)
        # 2. Transform old nodes to new nodes
        self.play(Transform(layer_group, new_layer_group)
                  , run_time=self.ANIMATION_RUN_TIME)

    def animate_connections(self, left_layer_centers, right_layer_centers
                            , line_group, w):
        # 1. Create connections with new parameters
        new_line_group = self.create_connections(left_layer_centers
                                                 , right_layer_centers
                                                 , w)
        # 2. Transform old connections to new connections
        self.play(Transform(line_group, new_line_group)
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

    def animate_prediction_text(self, prediction_text_group, prediction, left_shift):
        # 1. Create prediction text with new parameters
        new_prediction_text_group = self.create_prediction_text(prediction
                                                                , left_shift)
        # 2. Transform old prediction text to new prediction text
        self.play(Transform(prediction_text_group, new_prediction_text_group)
                  , run_time=self.ANIMATION_RUN_TIME)
        # self.play(Circumscribe(prediction_text, Circle))








