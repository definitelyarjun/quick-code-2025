import React, { useState, useEffect } from 'react';
import { Calendar as CalendarIcon, Moon, Sun } from 'lucide-react';
import { StudyPlanner } from './components/StudyPlanner';
import { StudyQuerySystem } from './components/StudyQuerySystem';
import { api } from './services/api';

function App() {
  const [activeSection, setActiveSection] = useState('query');
  const [isDarkMode, setIsDarkMode] = useState(false);
  
  // Apply theme class to body
  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark');
    } else {
      document.body.classList.remove('dark');
    }
  }, [isDarkMode]);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };
  
  const handleSaveStudyPlan = async (data) => {
    try {
      // Upload PDFs first
      const uploadPromises = data.files.map(file => api.uploadPDF(file));
      await Promise.all(uploadPromises);

      // Generate schedule
      const scheduleData = {
        test_date: data.testDate.toISOString().split('T')[0],
        completed_chapters: data.chapters
          .filter(ch => ch.completed)
          .map(ch => ({
            chapter: ch.name,
            time_taken: ch.timeSpent / 60, // Convert minutes to hours
            completion_percentage: 100,
          })),
        remaining_chapters: data.chapters
          .filter(ch => !ch.completed)
          .map(ch => ch.name),
      };

      const schedule = await api.generateSchedule(scheduleData);
      
      // Show schedule to user (you could add a state to display this)
      console.log('Generated Schedule:', schedule);
    } catch (error) {
      console.error('Error saving study plan:', error);
      // Handle error (show notification, etc.)
    }
  };

  const handleQuery = async (query) => {
    try {
      const results = await api.queryDocuments(query);
      return results;
    } catch (error) {
      console.error('Error querying documents:', error);
      return { error: 'Failed to fetch results' };
    }
  };

  return (
    <div className={`min-h-screen transition-colors duration-200 ${isDarkMode ? 'bg-gray-900 text-gray-100' : 'bg-white text-gray-900'}`}>
      <div className="container mx-auto px-4 py-12 max-w-5xl">
        {/* Header with Calendar Icon and Theme Toggle */}
        <div className="relative mb-8">
          <button 
            onClick={() => setActiveSection('planner')}
            className={`absolute left-0 top-1/2 -translate-y-1/2 p-3 rounded-full ${
              isDarkMode 
                ? 'bg-gray-800 hover:bg-gray-700 text-gray-300' 
                : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
            } transition-colors`}
            aria-label="Go to Planner"
          >
            <CalendarIcon className="h-6 w-6" />
          </button>
          
          {/* Theme Toggle Button */}
          <button 
            onClick={toggleTheme}
            className={`absolute right-0 top-1/2 -translate-y-1/2 p-3 rounded-full ${
              isDarkMode 
                ? 'bg-gray-800 hover:bg-gray-700 text-yellow-300' 
                : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
            } transition-colors`}
            aria-label={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {isDarkMode ? (
              <Sun className="h-6 w-6" />
            ) : (
              <Moon className="h-6 w-6" />
            )}
          </button>
          
          {/* Greeting */}
          <div className="text-center">
            <h1 className={`text-4xl font-semibold mb-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              Good afternoon
            </h1>
            <p className={`text-xl ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              How can I help you today?
            </p>
          </div>
        </div>

        {/* Main container - larger with rounded corners */}
        <div className={`rounded-3xl shadow-md border p-10 ${
          isDarkMode 
            ? 'bg-gray-800 border-gray-700' 
            : 'bg-gray-50 border-gray-200'
        }`}>
          {activeSection === 'planner' ? (
            <div>
              <div className="flex items-center mb-6">
                <h2 className={`text-2xl font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                  Study Planner
                </h2>
                <button 
                  onClick={() => setActiveSection('query')}
                  className={`ml-auto px-4 py-2 rounded-lg transition-colors ${
                    isDarkMode 
                      ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' 
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  Back to Query
                </button>
              </div>
              <StudyPlanner onSave={handleSaveStudyPlan} isDarkMode={isDarkMode} />
            </div>
          ) : (
            <StudyQuerySystem onQuery={handleQuery} isDarkMode={isDarkMode} />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
