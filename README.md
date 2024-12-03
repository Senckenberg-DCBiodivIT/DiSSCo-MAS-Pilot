# Plant Organ Segmentation Machine Annotation Service

This repository provides a machine annotation service for plant organ segmentation, designed to facilitate the extraction of additional information from herbarium sheets. It includes a trained model checkpoint and the necessary scripts to deploy and test the system.

---

## Install Dependencies

To set up the environment, follow these steps:

1. **Install required dependencies for the service**:

    ```bash
    pip install -r requirements_service.txt
    pip install -r requirements_inference.txt
    ```

---

## Usage

### Uvicorn Service

To start the server using `uvicorn`, run the `service.py` script. This will initialize the server and allow it to handle incoming requests for plant organ segmentation.

  Start the service:
  ```bash
  python service.py
  ```

### WebSocket

The `uvicorn` service forwards incoming requests to the WebSocket, where inference is performed and additional information such as surface area and other plant organ properties are extracted. To run the WebSocket process and handle inference, execute the following command:
  
  Start the websocket:
  ```bash
  python inference.py
  ```

### Standalone Testing

To test the service with images, the `test.py` script performs inference on the images located in the `test_image/scans` directory and outputs the results.

Run the standalone test:
  ```bash
   python test.py
  ```


