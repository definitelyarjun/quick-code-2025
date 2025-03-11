import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { StudyPlanner } from './components/StudyPlanner';
import { StudyQuerySystem } from './components/StudyQuerySystem';
import { api } from './services/api';

function Layout({ children }) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-grid-slate-200/50 bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_80%_80%_at_50%_50%,#000_20%,transparent_120%)]"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="h-[50rem] w-[50rem] rounded-full bg-gradient-to-tr from-purple-100/40 to-indigo-100/40 blur-3xl"></div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="relative z-10 border-b bg-white/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
              Study Assistant
            </h1>
            <div className="space-x-6">
              <Link to="/" className="text-gray-700 hover:text-indigo-600 transition-colors font-medium">
                Planner
              </Link>
              <Link to="/query" className="text-gray-700 hover:text-indigo-600 transition-colors font-medium">
                Study Query
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 relative z-10">
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-gray-100">
          {children}
        </div>
      </main>
    </div>
  );
}

function PlannerPage() {
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

  return (
    <Layout>
      <StudyPlanner onSave={handleSaveStudyPlan} />
    </Layout>
  );
}

function QueryPage() {
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
    <Layout>
      <StudyQuerySystem onQuery={handleQuery} />
    </Layout>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<PlannerPage />} />
        <Route path="/query" element={<QueryPage />} />
      </Routes>
    </Router>
  );
}

export default App;
