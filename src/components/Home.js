import React, { useState } from 'react';
import axios from 'axios';
import exampleMRI from '../brainimage.jpeg'; // Ensure the path is correct
import './Home.css'; // Import the CSS file for styles

const Home = () => {
  const [file, setFile] = useState(null);
  const [imageURL, setImageURL] = useState(exampleMRI); // Default to the home image
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [prediction, setPrediction] = useState('');
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type.includes('image')) {
      setFile(selectedFile);
      setImageURL(URL.createObjectURL(selectedFile)); // Create a URL for the uploaded image
      setError(''); // Clear previous errors on new valid file selection
    } else {
      setFile(null);
      setImageURL(exampleMRI); // Reset to home image for invalid file
      setError('Please upload a valid MRI scan image.'); // Show error for invalid file
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please upload a file before submitting.');
      return;
    }

    setIsSubmitting(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5001/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data.prediction);
      setError(''); // Clear any error messages on success
    } catch (error) {
      console.error('Error during prediction:', error);
      setError('Error during prediction. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="home-container">
      {/* Left section: Upload box */}
      <div className="upload-section">
        {/* Welcome note section */}
        <div className="welcome-note">
          <h2>Welcome to the MRI Checking and Reporting System</h2>
          <p>Upload your MRI scan to receive instant analysis and reporting.</p>
        </div>

        <div className="upload-box">
          {/* Removed heading */}
          <button onClick={() => document.getElementById('file-input').click()}>Select MRI Scans</button>
          <input
            id="file-input"
            type="file"
            onChange={handleFileChange}
            accept="image/*"
            style={{ display: 'none' }} // Hide the file input
          />
          <button onClick={handleSubmit} disabled={isSubmitting}>Submit</button>
          {error && <p className="error">{error}</p>}
          {prediction && <p className="prediction">Prediction: {prediction}</p>}
        </div>
      </div>

      {/* Right section: Image and quote */}
      <div className="right-section">
        {/* Image container */}
        <div className="image-container">
          <img src={imageURL} alt="Uploaded MRI" className="uploaded-image" />
        </div>

        {/* Quote section */}
        <div className="quote">
          “The brain is a wonderful organ. It starts working the moment you get up in the morning and does not stop until you get into the office.” - Robert Frost
        </div>
      </div>
    </div>
  );
};

export default Home;

