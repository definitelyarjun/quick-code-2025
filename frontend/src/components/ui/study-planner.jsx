import React, { useState } from "react";
import { Calendar as CalendarIcon, Plus, X, Edit, Trash2, Clock } from "lucide-react";
import { format, startOfMonth, endOfMonth, eachDayOfInterval, isSameDay, isToday, addMonths, subMonths } from "date-fns";
import { cn } from "../../lib/utils";

export function StudyPlanner({ onSave, isDarkMode = false }) {
  const [currentMonth, setCurrentMonth] = useState(new Date());
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [tasks, setTasks] = useState([
    { id: 1, date: new Date(), title: "Review Chapter 1", time: "10:00 AM" },
    { id: 2, date: new Date(new Date().setDate(new Date().getDate() + 2)), title: "Complete Practice Test", time: "2:00 PM" },
    { id: 3, date: new Date(new Date().setDate(new Date().getDate() + 5)), title: "Study Group Meeting", time: "4:30 PM" },
  ]);
  const [isAddingTask, setIsAddingTask] = useState(false);
  const [isEditingTask, setIsEditingTask] = useState(null);
  const [newTaskTitle, setNewTaskTitle] = useState("");
  const [newTaskTime, setNewTaskTime] = useState("");

  // Generate days for the current month view
  const monthStart = startOfMonth(currentMonth);
  const monthEnd = endOfMonth(currentMonth);
  const daysInMonth = eachDayOfInterval({ start: monthStart, end: monthEnd });

  // Get day names for header
  const weekDays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

  // Get tasks for selected date
  const selectedDateTasks = tasks.filter(task => 
    isSameDay(new Date(task.date), selectedDate)
  );

  // Handle navigation between months
  const nextMonth = () => setCurrentMonth(addMonths(currentMonth, 1));
  const prevMonth = () => setCurrentMonth(subMonths(currentMonth, 1));

  // Handle adding a new task
  const handleAddTask = () => {
    if (newTaskTitle.trim() === "") return;
    
    const newTask = {
      id: Date.now(),
      date: selectedDate,
      title: newTaskTitle,
      time: newTaskTime || "All day"
    };
    
    setTasks([...tasks, newTask]);
    setNewTaskTitle("");
    setNewTaskTime("");
    setIsAddingTask(false);
  };

  // Handle updating a task
  const handleUpdateTask = () => {
    if (newTaskTitle.trim() === "" || !isEditingTask) return;
    
    setTasks(tasks.map(task => 
      task.id === isEditingTask.id 
        ? { ...task, title: newTaskTitle, time: newTaskTime || task.time } 
        : task
    ));
    
    setNewTaskTitle("");
    setNewTaskTime("");
    setIsEditingTask(null);
  };

  // Handle deleting a task
  const handleDeleteTask = (taskId) => {
    setTasks(tasks.filter(task => task.id !== taskId));
  };

  // Start editing a task
  const startEditTask = (task) => {
    setIsEditingTask(task);
    setNewTaskTitle(task.title);
    setNewTaskTime(task.time);
  };

  // Get task count for a specific day (for the dots in calendar)
  const getTaskCountForDay = (day) => {
    return tasks.filter(task => isSameDay(new Date(task.date), day)).length;
  };

  return (
    <div className="space-y-6">
      {/* Calendar Header */}
      <div className="flex items-center justify-between">
        <h2 className={`text-2xl font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
          {format(currentMonth, "MMMM yyyy")}
        </h2>
        <div className="flex space-x-2">
          <button 
            onClick={prevMonth}
            className={`p-2 rounded-lg ${
              isDarkMode 
                ? 'hover:bg-gray-700 text-gray-300' 
                : 'hover:bg-gray-100 text-gray-700'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          </button>
          <button 
            onClick={nextMonth}
            className={`p-2 rounded-lg ${
              isDarkMode 
                ? 'hover:bg-gray-700 text-gray-300' 
                : 'hover:bg-gray-100 text-gray-700'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
      </div>

      {/* Calendar Grid */}
      <div className={`rounded-xl overflow-hidden border ${
        isDarkMode ? 'border-zinc-800' : 'border-gray-200'
      }`}>
        {/* Day names header */}
        <div className={`grid grid-cols-7 ${
          isDarkMode ? 'bg-black' : 'bg-gray-50'
        }`}>
          {weekDays.map((day) => (
            <div 
              key={day} 
              className={`py-3 text-center text-sm font-medium ${
                isDarkMode ? 'text-zinc-300' : 'text-gray-500'
              }`}
            >
              {day}
            </div>
          ))}
        </div>
        
        {/* Calendar days */}
        <div className={`grid grid-cols-7 ${
          isDarkMode ? 'bg-black' : 'bg-white'
        }`}>
          {daysInMonth.map((day, i) => {
            const taskCount = getTaskCountForDay(day);
            return (
              <button
                key={i}
                onClick={() => setSelectedDate(day)}
                className={cn(
                  `h-24 p-2 border-t border-r relative transition-colors ${
                    isDarkMode 
                      ? 'border-zinc-800 hover:bg-zinc-900 text-zinc-300' 
                      : 'border-gray-100 hover:bg-gray-50 text-gray-700'
                  }`,
                  i % 7 === 6 && "border-r-0", // Remove right border on last column
                  isSameDay(day, selectedDate) && (isDarkMode ? 'bg-black border-zinc-700' : 'bg-blue-50'),
                  isToday(day) && "font-bold"
                )}
              >
                <span className={cn(
                  "inline-flex h-6 w-6 items-center justify-center rounded-full text-sm",
                  isToday(day) && (isDarkMode ? 'bg-black text-white border border-zinc-700' : 'bg-blue-100 text-blue-700')
                )}>
                  {format(day, "d")}
                </span>
                
                {/* Task indicators */}
                {taskCount > 0 && (
                  <div className="absolute bottom-2 right-2 flex space-x-1">
                    {taskCount <= 3 ? (
                      Array(taskCount).fill(0).map((_, i) => (
                        <div key={i} className={`h-2 w-2 rounded-full ${
                          isDarkMode ? 'bg-zinc-500' : 'bg-blue-500'
                        }`}></div>
                      ))
                    ) : (
                      <div className={`text-xs font-medium px-1.5 py-0.5 rounded-full ${
                        isDarkMode 
                          ? 'bg-black text-zinc-300 border border-zinc-800' 
                          : 'bg-blue-100 text-blue-600'
                      }`}>
                        {taskCount}
                      </div>
                    )}
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Selected Day Tasks */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className={`text-lg font-medium ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>
            {format(selectedDate, "MMMM d, yyyy")}
          </h3>
          {!isAddingTask && !isEditingTask && (
            <button
              onClick={() => setIsAddingTask(true)}
              className={`flex items-center gap-1 px-3 py-1.5 rounded-lg transition-colors ${
                isDarkMode 
                  ? 'bg-black hover:bg-zinc-900 text-zinc-300 border border-zinc-800' 
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
              }`}
            >
              <Plus className="h-4 w-4" />
              <span>Add Task</span>
            </button>
          )}
        </div>

        {/* Task list */}
        {selectedDateTasks.length === 0 && !isAddingTask && !isEditingTask ? (
          <div className={`py-8 text-center border-2 border-dashed rounded-xl ${
            isDarkMode 
              ? 'bg-black text-zinc-400 border-zinc-800' 
              : 'bg-gray-50 text-gray-500 border-gray-200'
          }`}>
            No tasks scheduled for this day
          </div>
        ) : (
          <div className="space-y-2">
            {selectedDateTasks.map(task => (
              <div 
                key={task.id}
                className={cn(
                  `p-4 rounded-xl border ${
                    isDarkMode 
                      ? 'bg-black border-zinc-800' 
                      : 'bg-white border-gray-200'
                  }`,
                  isEditingTask?.id === task.id && (isDarkMode ? 'ring-2 ring-zinc-700' : 'ring-2 ring-blue-200')
                )}
              >
                {isEditingTask?.id === task.id ? (
                  <div className="space-y-3">
                    <input
                      type="text"
                      value={newTaskTitle}
                      onChange={(e) => setNewTaskTitle(e.target.value)}
                      className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 ${
                        isDarkMode 
                          ? 'bg-black border-zinc-800 text-white focus:ring-zinc-700' 
                          : 'border-gray-300 focus:ring-blue-500'
                      }`}
                      placeholder="Task title"
                    />
                    <div className="flex items-center gap-2">
                      <Clock className={isDarkMode ? 'text-zinc-500' : 'text-gray-500'} />
                      <input
                        type="text"
                        value={newTaskTime}
                        onChange={(e) => setNewTaskTime(e.target.value)}
                        className={`flex-1 px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 ${
                          isDarkMode 
                            ? 'bg-black border-zinc-800 text-white focus:ring-zinc-700' 
                            : 'border-gray-300 focus:ring-blue-500'
                        }`}
                        placeholder="Time (e.g. 3:00 PM)"
                      />
                    </div>
                    <div className="flex justify-end space-x-2 pt-2">
                      <button
                        onClick={() => setIsEditingTask(null)}
                        className={`px-3 py-1.5 rounded-lg ${
                          isDarkMode 
                            ? 'bg-black hover:bg-zinc-900 text-zinc-300 border border-zinc-800' 
                            : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                        }`}
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleUpdateTask}
                        className={`px-3 py-1.5 rounded-lg ${
                          isDarkMode 
                            ? 'bg-black hover:bg-zinc-900 text-white border border-zinc-700' 
                            : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
                        }`}
                      >
                        Save
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className={`font-medium ${
                        isDarkMode ? 'text-zinc-200' : 'text-gray-900'
                      }`}>
                        {task.title}
                      </h4>
                      <div className={`flex items-center mt-1 text-sm ${
                        isDarkMode ? 'text-zinc-400' : 'text-gray-500'
                      }`}>
                        <Clock className="h-3.5 w-3.5 mr-1" />
                        {task.time}
                      </div>
                    </div>
                    <div className="flex space-x-1">
                      <button
                        onClick={() => startEditTask(task)}
                        className={`p-1.5 rounded-lg ${
                          isDarkMode 
                            ? 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900' 
                            : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                        }`}
                      >
                        <Edit className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => handleDeleteTask(task.id)}
                        className={`p-1.5 rounded-lg ${
                          isDarkMode 
                            ? 'text-zinc-400 hover:text-red-300 hover:bg-zinc-900' 
                            : 'text-gray-500 hover:text-red-600 hover:bg-red-50'
                        }`}
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Add new task form */}
            {isAddingTask && (
              <div className={`p-4 rounded-xl border ${
                isDarkMode 
                  ? 'bg-black border-zinc-800 ring-2 ring-zinc-700' 
                  : 'bg-white border-gray-200 ring-2 ring-blue-200'
              }`}>
                <div className="space-y-3">
                  <input
                    type="text"
                    value={newTaskTitle}
                    onChange={(e) => setNewTaskTitle(e.target.value)}
                    className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 ${
                      isDarkMode 
                        ? 'bg-black border-zinc-800 text-white focus:ring-zinc-700' 
                        : 'border-gray-300 focus:ring-blue-500'
                    }`}
                    placeholder="Task title"
                    autoFocus
                  />
                  <div className="flex items-center gap-2">
                    <Clock className={isDarkMode ? 'text-zinc-500' : 'text-gray-500'} />
                    <input
                      type="text"
                      value={newTaskTime}
                      onChange={(e) => setNewTaskTime(e.target.value)}
                      className={`flex-1 px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 ${
                        isDarkMode 
                          ? 'bg-black border-zinc-800 text-white focus:ring-zinc-700' 
                          : 'border-gray-300 focus:ring-blue-500'
                      }`}
                      placeholder="Time (e.g. 3:00 PM)"
                    />
                  </div>
                  <div className="flex justify-end space-x-2 pt-2">
                    <button
                      onClick={() => setIsAddingTask(false)}
                      className={`px-3 py-1.5 rounded-lg ${
                        isDarkMode 
                          ? 'bg-black hover:bg-zinc-900 text-zinc-300 border border-zinc-800' 
                          : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                      }`}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleAddTask}
                      className={`px-3 py-1.5 rounded-lg ${
                        isDarkMode 
                          ? 'bg-black hover:bg-zinc-900 text-white border border-zinc-700' 
                          : 'bg-blue-100 hover:bg-blue-200 text-blue-700'
                      }`}
                    >
                      Add Task
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
