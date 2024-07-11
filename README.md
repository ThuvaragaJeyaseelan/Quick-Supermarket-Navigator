# Quick-Supermarket-Navigator

Quick-Supermarket-Navigator is a web application designed to optimize your shopping experience by finding the shortest path through a supermarket based on your shopping list. The application uses OCR (Optical Character Recognition) to extract text from images of shopping lists, identifies the items, and calculates the shortest path to collect all items efficiently. 

## Features

- **OCR Integration**: Extracts text from images of shopping lists using PaddleOCR.
- **Keyword Extraction**: Identifies items from the extracted text using KeyBERT.
- **Spell Correction**: Corrects misspelled item names using SymSpell.
- **Fuzzy Matching**: Matches identified items to known supermarket locations using fuzzy matching.
- **Shortest Path Calculation**: Calculates the shortest path to collect all items using the Dijkstra algorithm implemented in NetworkX.
- **Visual Representation**: Displays the supermarket layout and the optimal path using Plotly.

## Technologies Used

- **Flask**: Web framework for Python to build the web application.
- **PaddleOCR**: Optical character recognition for extracting text from images.
- **KeyBERT**: Keyword extraction to identify items from text.
- **SymSpell**: Spell correction for identified items.
- **FuzzyWuzzy**: Fuzzy string matching for item names.
- **NetworkX**: Graph-based calculations for shortest path finding.
- **Plotly**: Visualization of the supermarket layout and path.
