import React from 'react';
import { StudyQuerySystem as StudyQueryUI } from './ui/study-query-system';

export function StudyQuerySystem({ onQuery }) {
  const handleQuery = async (query) => {
    if (onQuery) {
      return await onQuery(query);
    }
  };

  return <StudyQueryUI onQuery={handleQuery} />;
}
