from flask import Flask, request, render_template
import pygad
import numpy as np
import pygad.kerasga
import tensorflow as tf

app = Flask(__name__)
global model
global data_inputs

def fitness_func(solution, sol_idx):
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)
    predictions = model.predict(data_inputs)
    mae = tf.keras.losses.MeanAbsoluteError()
    solution_fitness = 1.0 / (mae(data_inputs, predictions).numpy() + 0.00000001)
    return solution_fitness

def callback_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

def predict_outcome(data_inputs):
    input_layer  = tf.keras.layers.Input(shape=(15,))
    dense_layer1 = tf.keras.layers.Dense(5, activation="relu")(input_layer)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(dense_layer1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    weights_vector = pygad.kerasga.model_weights_as_vector(model=model)
    keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)

    filename = "Final_Dataset"
    ga_instance = pygad.load(filename=filename)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)

    model.set_weights(best_solution_weights)
    predictions = model.predict(data_inputs)
    predictions = predictions[0][0] * 1000000 // 1

    return predictions

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/calculate', methods=["GET", "POST"])
def calculate():
    if request.method == "POST":
        try:
            area = np.float64(int(request.form.get("Area_Project", 0)) / 10000)
            no_floor = np.float64(request.form.get("No_floors", 0))
            parking = np.float64(request.form.get("No_parking", 0))
            duration = np.float64(request.form.get("Duration", 0))
            escalations = np.float64(request.form.get("Escalations", 0))
            change = np.float64(request.form.get("Changes_Range", 0))
            earthwork = np.float64(request.form.get("Earthwork", 0))
            foundation = request.form.get("Foundation", "Mat")
            external = request.form.get("External_wall", "Emulsion")
            ceiling = request.form.get("Ceiling", "Non_Asbestos")
            internal = request.form.get("Internal_wall", "Acrylic")
            form = request.form.get("Form_system", "Ganged")
            sub = request.form.get("Substructure", "RCC")
            super_structure = request.form.get("Superstructure", "RCC")
            floor = request.form.get("Floor", "Laminate")

            sub = np.float64(1) if sub == "RCC" else np.float64(0)
            super_structure = np.float64(1) if super_structure == "RCC" else np.float64(0)
            ceiling = np.float64(1) if ceiling == "Non_Asbestos" else np.float64(2)
            form = np.float64(1) if form == "Ganged" else np.float64(2)
            external = {"Oil": np.float64(1), "Enamel": np.float64(2), "Emulsion": np.float64(3)}.get(external, np.float64(0))
            foundation = {"Mat": np.float64(1), "Wall Footing": np.float64(2), "Pile": np.float64(3)}.get(foundation, np.float64(0))
            internal = {"Acrylic": np.float64(1), "Emulsion": np.float64(2), "Gypsum_Board": np.float64(3)}.get(internal, np.float64(0))
            floor = {"Marble": np.float64(1), "Vinyl": np.float64(2), "Laminate": np.float64(3)}.get(floor, np.float64(0))

            data_inputs = [[area, ceiling, external, internal, floor, foundation, form, super_structure, sub, change, duration, earthwork, no_floor, parking, escalations]]

            print(data_inputs)

            sum = predict_outcome(data_inputs)

            final_output = f"Predicted output based on the best solution: {sum}"
        except Exception as e:
            final_output = f"An error occurred: {str(e)}"
            print(f"Error: {str(e)}")

    return render_template("main.html", predict_content=final_output)

if __name__ == '__main__':
    app.run(debug=True)
