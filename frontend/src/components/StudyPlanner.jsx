import React, { useState } from 'react';
import { StudyPlanner as StudyPlannerUI } from './ui/study-planner';

export function StudyPlanner({ onSave }) {
  const handleSaveStudyPlan = async (data) => {
    if (onSave) {
      await onSave(data);
    }
  };

  return <StudyPlannerUI onSave={handleSaveStudyPlan} />;
}
