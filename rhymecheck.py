import random
import math

# BASIC MATH HELPERS

def dot(v1, v2):
    total = 0
    for i in range(len(v1)):
        total += v1[i] * v2[i]
    return total

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# WORD ENCODING

vowel_map = {'a':0.2,'e':0.4,'i':0.6,'o':0.8,'u':1.0}

def encode_word(word):
    encoded = []
    for char in word[-4:]:           # last 4 letters matter
        if char in vowel_map:
            encoded.append(vowel_map[char])
        elif char.isalpha():
            encoded.append(0.2)    # consonant
    while len(encoded) < 4:
        encoded.insert(0, 0)
    return encoded

# DATASET
# label=1 = Rhyme
# label=0 = Not Rhyme

data = [
    ("stand", "band", 1),
    ("time", "rhyme", 1),
    ("mat", "hat", 1),
    ("stand", "stone", 0),
    ("blue", "pizza", 0),
    ("sing", "ring", 1),
    ("sky", "moon", 0),
    ("fun", "run", 1),
    ("run", "bun", 1),
]

# INPUT VECTOR

def make_input(word1, word2):
    return encode_word(word1.lower()) + encode_word(word2.lower())

# NETWORK INITIALIZATION

random.seed(1)

input_size = 8
hidden_size = 4
learning_rate = 0.5

w1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
b1 = [0] * hidden_size

w2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
b2 = 0

# FORWARD PASS

def forward(x):
    hidden = []
    for i in range(hidden_size):
        z = dot(w1[i], x) + b1[i]
        hidden.append(sigmoid(z))

    output = dot(w2, hidden) + b2
    output = sigmoid(output)

    return hidden, output

# TRAINING LOOP (BACKPROP)

for epoch in range(500):
    total_loss = 0

    for word1, word2, label in data:
        x = make_input(word1,word2)
        hidden, output = forward(x)

        # loss (MSE)
        loss = (output - label) ** 2
        total_loss += loss

        # output gradient
        d_output = 2 * (output - label) * output * (1 - output)

        # update output weights
        for i in range(hidden_size):
            w2[i] -= learning_rate * d_output * hidden[i]
        b2 -= learning_rate * d_output

        # update hidden weights
        for i in range(hidden_size):
            d_hidden = d_output * w2[i] * hidden[i] * (1 - hidden[i])
            for j in range(input_size):
                w1[i][j] -= learning_rate * d_hidden * x[j]
            b1[i] -= learning_rate * d_hidden

    if epoch % 100 == 0:
        print("Epoch", epoch, "Loss", round(total_loss,4))

print("\nHIDDEN LAYER WEIGHTS (w1):")
for i in range(hidden_size):
    print(f"Neuron {i+1} weights:", [round(w, 3) for w in w1[i]])
    print(f"Neuron {i+1} bias:", round(b1[i], 3))

print("\nOUTPUT LAYER WEIGHTS (w2):")
print("Output weights:", [round(w, 3) for w in w2])
print("Output bias:", round(b2, 3))

# PREDICTION

def predict(word1, word2):
    x = make_input(word1, word2)
    _, output = forward(x)

    print("confidence:", round(output*100, 3))

    if output < 0.2:
        result = "label=0 Not Rhyme"
        print("Very unlikely to rhyme")
    elif output < 0.6:
        result = "label=0 Not Rhyme"
        print("Possibly not a rhyme")
    else:
        result = "label=1 Rhyme"
        print("Sounds like a rhyme")

    return result

# TEST

print(predict("bat", "hat"))
print(predict("fly", "moon"))

# USER INPUT TEST LOOP

boolean_value = True

while boolean_value:
  user_word1 = input("Enter word 1: ")
  user_word2 = input("Enter word 2: ")
  print("Prediction:", predict(user_word1, user_word2))

  user_choice = input("Want to continue? (Y/N)\n").lower()

  if user_choice == "y":
    continue
  else:
    boolean_value = False
