# Sentry: Guardian Angel for Machinery

Sentry is an innovative application designed to revolutionize machinery maintenance by offering defect detection and failure prediction capabilities. With Sentry, businesses can proactively identify issues, minimize downtime, and enhance operational efficiency.

## Features

- **Defect Detection**: Utilizes cutting-edge machine learning algorithms to detect defects in machinery components, providing precise bounding box labels for different defect types.
- **Failure Prediction**: Predicts machinery failures based on various operational parameters, enabling proactive maintenance and minimizing disruptions.
- **User-Friendly Interface**: Seamlessly integrates both defect detection and failure prediction functionalities into a user-friendly interface, facilitating easy interaction and interpretation of results.

## Technical Overview

- **Defect Detection**: Implements YOLOv8 for defect detection, trained on a curated dataset of PCB images with bounding box labels. Utilizes Flask for processing uploaded images and Docker for seamless deployment.
- **Failure Prediction**: Utilizes XGBoost for failure prediction, trained on milling machine data with extensive preprocessing and feature engineering. Deployed within Docker, accepting JSON input data for efficient predictions.
- **Business Interface**: Integrates defect detection and failure prediction functionalities via HTML templates, orchestrated using Docker Compose for streamlined deployment.

## Getting Started

To get started with Sentry, follow these steps:

1. **Clone the Repository**: 'git clone https://github.com/Mariamhsein2001/Industrial_AI'
2. **Pull Images**: 'docker pull mariamhsein/manufacturing_ai:prediciton_failure' /  'docker pull mariamhsein/manufacturing_ai:businesses' / 'docker pull mariamhsein/manufacturing_ai:detection'
3. **Run the Application**: Execute 'kubectl apply -f k8s-deployment.yml' to start the application. Access the interface via `http://localhost:30500` in your web browser.


## Support

If you encounter any issues or have questions about Sentry, please [open an issue](https://github.com/Mariamhsein2001/Industrial_AI/issues) on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
