from flask import Flask, request, render_template, redirect, url_for
import networkx as nx
from keybert import KeyBERT
from itertools import permutations
import os
from symspellpy import SymSpell, Verbosity
from fuzzywuzzy import process
import plotly.graph_objects as go
from paddleocr import PaddleOCR
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Create a graph representing the supermarket layout
G = nx.Graph()

# Define nodes and their positions (for visualization purposes)
nodes = {
    'Entrance': (0, 0),
    'Aisle 1': (1, 0),
    'Aisle 2': (2, 0),
    'Aisle 3': (3, 0),
    'Aisle 4': (4, 0),
    'Aisle 5': (5, 0),
    'Dairy': (1, 1),
    'Bakery': (2, 1),
    'Produce': (3, 1),
    'Meat': (4, 1),
    'Frozen': (5, 1),
    'Snacks': (1, 2),
    'Beverages': (2, 2),
    'Household': (3, 2),
    'Personal Care': (4, 2),
    'Checkout': (5, 2),
    'Corner 1': (1, -1),
    'Corner 2': (2, -1),
    'Corner 3': (3, -1),
    'Corner 4': (4, -1),
    'Corner 5': (5, -1),
    'Corner 6': (1, 3),
    'Corner 7': (2, 3),
    'Corner 8': (3, 3),
    'Corner 9': (4, 3),
    'Corner 10': (5, 3)
}

# Add nodes to the graph
for node, pos in nodes.items():
    G.add_node(node, pos=pos)

# Add edges with distances
edges = [
    ('Entrance', 'Aisle 1', 1),
    ('Aisle 1', 'Aisle 2', 1),
    ('Aisle 2', 'Aisle 3', 1),
    ('Aisle 3', 'Aisle 4', 1),
    ('Aisle 4', 'Aisle 5', 1),
    ('Aisle 1', 'Dairy', 1),
    ('Aisle 2', 'Bakery', 1),
    ('Aisle 3', 'Produce', 1),
    ('Aisle 4', 'Meat', 1),
    ('Aisle 5', 'Frozen', 1),
    ('Dairy', 'Snacks', 1),
    ('Bakery', 'Beverages', 1),
    ('Produce', 'Household', 1),
    ('Meat', 'Personal Care', 1),
    ('Frozen', 'Checkout', 1),
    ('Corner 1', 'Dairy', 1),
    ('Corner 2', 'Bakery', 1),
    ('Corner 3', 'Produce', 1),
    ('Corner 4', 'Meat', 1),
    ('Corner 5', 'Frozen', 1),
    ('Corner 1', 'Corner 2', 1),
    ('Corner 2', 'Corner 3', 1),
    ('Corner 3', 'Corner 4', 1),
    ('Corner 4', 'Corner 5', 1),
    ('Corner 6', 'Snacks', 1),
    ('Corner 7', 'Beverages', 1),
    ('Corner 8', 'Household', 1),
    ('Corner 9', 'Personal Care', 1),
    ('Corner 10', 'Checkout', 1),
    ('Corner 6', 'Corner 7', 1),
    ('Corner 7', 'Corner 8', 1),
    ('Corner 8', 'Corner 9', 1),
    ('Corner 9', 'Corner 10', 1)
]

for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])

# Initialize KeyBERT model
kw_model = KeyBERT()

# Initialize SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = os.path.join(os.path.dirname(__file__), 'items_dictionary.txt')
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Improved item location mapping with more comprehensive data
item_locations = {
    'milk': 'Dairy',
    'bread': 'Bakery',
    'apple': 'Produce',
    'chicken': 'Meat',
    'ice cream': 'Frozen',
    'chips': 'Snacks',
    'soda': 'Beverages',
    'detergent': 'Household',
    'shampoo': 'Personal Care',
    'butter': 'Dairy',
    'cake': 'Bakery',
    'banana': 'Produce',
    'beef': 'Meat',
    'frozen pizza': 'Frozen',
    'cookies': 'Snacks',
    'juice': 'Beverages',
    'cleaner': 'Household',
    'toothpaste': 'Personal Care',
    'yogurt': 'Dairy',
    'muffins': 'Bakery',
    'grapes': 'Produce',
    'pork': 'Meat',
    'frozen veggies': 'Frozen',
    'crackers': 'Snacks',
    'water': 'Beverages',
    'laundry detergent': 'Household',
    'conditioner': 'Personal Care',
    'cheese': 'Dairy',
    'bread rolls': 'Bakery',
    'oranges': 'Produce',
    'turkey': 'Meat',
    'frozen fruit': 'Frozen',
    'granola bars': 'Snacks',
    'coffee': 'Beverages',
    'dish soap': 'Household',
    'body wash': 'Personal Care'
}


def get_item_location(item_name):
    return item_locations.get(item_name.lower(), 'Unknown')


def correct_item_name(item_name):
    suggestions = sym_spell.lookup(item_name, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    return item_name


def fuzzy_match_item(item_name, item_list, threshold=80):
    match, score = process.extractOne(item_name, item_list)
    if score >= threshold:
        return match
    return None


def predict_item_names(input_text):
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in input_text.split() if word.lower() not in stop_words]

    # Use KeyBERT to extract keywords
    filtered_text = ' '.join(filtered_words)
    keywords = kw_model.extract_keywords(filtered_text, keyphrase_ngram_range=(1, 1), stop_words=None)
    item_names = [kw[0] for kw in keywords]

    # Print debug information
    print("Extracted Keywords:", item_names)

    corrected_item_names = []
    for item in item_names:
        corrected_item = correct_item_name(item)
        corrected_item_names.append(corrected_item)

    # Print debug information
    print("Corrected Item Names:", corrected_item_names)

    fuzzy_matched_items = []
    for item in corrected_item_names:
        matched_item = fuzzy_match_item(item, item_locations.keys())
        fuzzy_matched_items.append(matched_item)

    # Print debug information
    print("Fuzzy Matched Items:", fuzzy_matched_items)

    return [item for item in fuzzy_matched_items if item]


def find_shortest_path(graph, start, end):
    return nx.shortest_path(graph, source=start, target=end, weight='weight')


def find_shortest_path_multiple(graph, start, items):
    min_path = None
    min_distance = float('inf')

    for perm in permutations(items):
        current_distance = 0
        current_path = [start]

        for i in range(len(perm)):
            if i == 0:
                partial_path = find_shortest_path(graph, start, perm[i])
            else:
                partial_path = find_shortest_path(graph, perm[i - 1], perm[i])
                partial_path = partial_path[1:]  # Avoid duplicating nodes
            current_path.extend(partial_path)
            current_distance += sum(nx.dijkstra_path_length(graph, partial_path[j], partial_path[j + 1]) for j in
                                    range(len(partial_path) - 1))

        if current_distance < min_distance:
            min_distance = current_distance
            min_path = current_path

    # Remove duplicate 'Entrance' if it appears more than once
    if min_path and len(min_path) > 1 and min_path[0] == start and min_path[1] == start:
        min_path = min_path[1:]

    # Add the path to the checkout from the last item
    if min_path:
        checkout_path = find_shortest_path(graph, min_path[-1], 'Checkout')
        min_path.extend(checkout_path[1:])  # Skip the starting node to avoid duplication

    return min_path, min_distance


def visualize_path_with_plotly(graph, pos, path, title):
    fig = go.Figure()

    # Draw nodes
    for node, (x, y) in pos.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            text=[node], textposition="top center",
            mode='markers+text',
            marker=dict(size=15, color='skyblue'),
            showlegend=False
        ))

    # Draw edges
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=2, color='lightgray'),
            showlegend=False
        ))

    # Define colors for subparts
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Highlight subparts of the path with different colors and arrows
    path_edges = list(zip(path, path[1:]))
    for i, edge in enumerate(path_edges):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines+markers',
            line=dict(width=4, color=colors[i % len(colors)]),
            marker=dict(size=10, color=colors[i % len(colors)]),
            showlegend=False
        ))
        # Add arrow annotation
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            axref='x', ayref='y',
            ax=(2 * x0 + x1) / 3,
            ay=(2 * y0 + y1) / 3,
            showarrow=True,
            arrowhead=3,
            arrowsize=2,
            arrowwidth=2,
            arrowcolor=colors[i % len(colors)]
        )

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig.to_json()


def extract_text_from_image(image_path):
    result = ocr.ocr(image_path, cls=True)
    text_lines = [line[1][0] for line in result[0]]
    return ' '.join(text_lines)


def generate_nlp_path_description(path, items):
    descriptions = []
    item_indices = {item: i for i, item in enumerate(path) if item in items}

    for i, item in enumerate(items):
        location = path[item_indices[item]]
        if i == 0:
            descriptions.append(f"First, enter and go to {location} to get the {item}.")
        else:
            prev_location = path[item_indices[items[i - 1]]]
            descriptions.append(f"Then, return to {prev_location} and go to {location} to get the {item}.")

    descriptions.append(f"Finally, go to the checkout.")

    return ' '.join(descriptions)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/find_path', methods=['POST'])
def find_path():
    input_text = request.form['items']
    item_names = predict_item_names(input_text)
    item_locations_list = [get_item_location(item) for item in item_names]

    if 'Unknown' not in item_locations_list:
        shortest_path, _ = find_shortest_path_multiple(G, 'Entrance', item_locations_list)
        plot_json = visualize_path_with_plotly(G, nx.get_node_attributes(G, 'pos'), shortest_path,
                                               f'Shortest Path to {item_names}')
        path_description = generate_nlp_path_description(shortest_path, item_locations_list)
        return render_template('result.html', item_names=item_names, plot_json=plot_json, path_text=path_description)
    else:
        return 'One or more items not found', 400


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        # Perform OCR on the image
        text = extract_text_from_image(filename)
        print("OCR Extracted Text:", text)  # Debugging
        # Use NLP to predict item names from the extracted text
        item_names = predict_item_names(text)
        print("Predicted Item Names:", item_names)  # Debugging
        item_locations_list = [get_item_location(item) for item in item_names]

        if 'Unknown' not in item_locations_list:
            shortest_path, _ = find_shortest_path_multiple(G, 'Entrance', item_locations_list)
            plot_json = visualize_path_with_plotly(G, nx.get_node_attributes(G, 'pos'), shortest_path,
                                                   f'Shortest Path to {item_names}')
            path_description = generate_nlp_path_description(shortest_path, item_locations_list)
            return render_template('result.html', item_names=item_names, plot_json=plot_json,
                                   path_text=path_description)
        else:
            return 'One or more items not found', 400
    return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)
