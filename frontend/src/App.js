import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { StudyPlanner } from './components/StudyPlanner';
import { StudyQuerySystem } from './components/StudyQuerySystem';
import { api } from './services/api';

function Layout({ children }) {
  return (
    <div className="min-h-screen bg-background">
      <nav className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-bold">Study Assistant</h1>
            <div className="space-x-4">
              <Link to="/" className="text-foreground hover:text-primary">Planner</Link>
              <Link to="/query" className="text-foreground hover:text-primary">Study Query</Link>
            </div>
          </div>
        </div>
      </nav>
      <main className="container mx-auto px-4 py-8">
        {children}
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
