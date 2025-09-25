# ids-housing-data

## Installation

### Backend (Python)
1. Navigate to the backend directory:
	```sh
	cd housingdata/src/backend
	```
2. (Optional) Create and activate a virtual environment:
	```sh
	python3 -m venv .venv
	source .venv/bin/activate
	```
3. Install dependencies:
	```sh
	pip install flask flask-cors
	```

### Frontend (Node.js)
1. Navigate to `housingdata/` and run:
	```sh
	npm install
	```

## Running the Project

### Start Backend Server
From `housingdata/src/backend`:
```sh
python app.py
```
or if using a virtual environment:
```sh
.venv/bin/python app.py
```

### Start Frontend Server
From `housingdata/`:
```sh
npm run dev
```