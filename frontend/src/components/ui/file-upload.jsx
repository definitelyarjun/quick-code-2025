import React, { useState } from 'react';
import { Upload, X, FileText, Check } from 'lucide-react';
import { Button } from './button';

export function FileUpload({ onUpload, isDarkMode }) {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError(null);
    } else {
      setFile(null);
      setError('Please select a valid PDF file');
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setError(null);
    
    try {
      await onUpload(file);
      setUploadSuccess(true);
      setTimeout(() => {
        setUploadSuccess(false);
        setFile(null);
      }, 3000);
    } catch (err) {
      setError('Failed to upload file. Please try again.');
      console.error('Upload error:', err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === 'application/pdf') {
      setFile(droppedFile);
      setError(null);
    } else {
      setError('Please drop a valid PDF file');
    }
  };

  return (
    <div className={`w-full ${isDarkMode ? 'text-gray-200' : 'text-gray-700'}`}>
      <div
        className={`border-2 border-dashed rounded-lg p-6 text-center ${
          isDarkMode 
            ? 'border-gray-600 hover:border-gray-500' 
            : 'border-gray-300 hover:border-gray-400'
        } transition-colors`}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center justify-center space-y-3">
          {!file ? (
            <>
              <Upload className={`h-10 w-10 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`} />
              <p className="text-sm">
                Drag and drop your PDF here, or{' '}
                <label className={`cursor-pointer ${isDarkMode ? 'text-blue-400 hover:text-blue-300' : 'text-blue-600 hover:text-blue-700'}`}>
                  browse
                  <input
                    type="file"
                    className="hidden"
                    accept=".pdf"
                    onChange={handleFileChange}
                  />
                </label>
              </p>
              <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Supports PDF files up to 10MB
              </p>
            </>
          ) : (
            <div className="w-full">
              <div className={`flex items-center p-3 rounded-md ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                <FileText className="h-6 w-6 mr-2" />
                <div className="flex-1 truncate">{file.name}</div>
                <button
                  onClick={() => setFile(null)}
                  className={`ml-2 p-1 rounded-full ${
                    isDarkMode 
                      ? 'hover:bg-gray-600 text-gray-300' 
                      : 'hover:bg-gray-200 text-gray-500'
                  }`}
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              
              <div className="mt-4 flex justify-center">
                <Button
                  onClick={handleUpload}
                  disabled={isUploading || uploadSuccess}
                  className={`px-4 py-2 rounded-md ${
                    isDarkMode
                      ? 'bg-blue-600 hover:bg-blue-700 text-white'
                      : 'bg-blue-500 hover:bg-blue-600 text-white'
                  }`}
                >
                  {isUploading ? (
                    <span className="flex items-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Uploading...
                    </span>
                  ) : uploadSuccess ? (
                    <span className="flex items-center">
                      <Check className="mr-2 h-4 w-4" />
                      Uploaded!
                    </span>
                  ) : (
                    'Upload PDF'
                  )}
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {error && (
        <div className={`mt-2 text-sm ${isDarkMode ? 'text-red-400' : 'text-red-500'}`}>
          {error}
        </div>
      )}
      
      {uploadSuccess && (
        <div className={`mt-2 text-sm ${isDarkMode ? 'text-green-400' : 'text-green-500'} flex items-center`}>
          <Check className="mr-1 h-4 w-4" />
          PDF uploaded successfully! You can now ask questions about it.
        </div>
      )}
    </div>
  );
} 