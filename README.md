# Movie_Recommendation

## Overview
This project is a movie recommendation system that suggests movies based on user preferences and viewing history.

## Contributors
| Name            | Student ID | Mail                          |
|-----------------|------------|-------------------------------|
| Chu Anh Duc     | 20230081   | duc.ca2030081@sis.hust.edu.vn |
| Vu Thuong Tin   | 20200091   | tin.vt200091@sis.hust.edu.vn  |
| Tran Quang Hung | 20235502   | hung.tq2035502@sis.hust.edu.vn|
| Phan Dinh Trung | 20230093   | trung.pd2030093@sis.hust.edu.vn|

## Folder Structure
```
Movie_Recommendation/
│
├── data/
│   ├── ml-100k/
│   │   ├── u.item
│   │   ├── ua.base
│   │   ├── ua.test
│   │   ├── ub.base
│   │   ├── ub.test
│   │   ├── ...
│   │   └── README
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── matrix-factorization-SVD.ipynb
│   ├── content_based.ipynb
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── lib/
│   │   ├── __init__.py
│   │   ├── collaborative_filtering.py
│   │   ├── evaluate.py
│   │   └── get_items_rated.py
│   ├── recommend_app_svd.py
│   ├── web.py
│   └── new_web.py
│
├── .gitignore
├── README.md
└── requirements.txt
```


## Features
- Personalized movie recommendations
- User-friendly interface
- Integration with popular movie databases

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Duckduck-05/Movie_Recommendation.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Movie_Recommendation
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the application:
    ```bash
    streamlit run src/new_web.py
    ```
2. Follow the on-screen instructions to get movie recommendations.


## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
