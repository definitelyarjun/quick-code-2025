import React, { useState, useRef } from "react";
import { Calendar as CalendarIcon, Clock, FileUp, Upload, X, CheckCircle, BookOpen, BookMarked } from "lucide-react";
import { cn } from "../../lib/utils";
import { Button } from "./button";
import { Input } from "./input";
import { Label } from "./label";
import { Progress } from "./progress";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./tabs";
import { Calendar } from "./calendar";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";
import { addDays, format } from "date-fns";

const presets = [
  { label: "Next Week", value: addDays(new Date(), 7) },
  { label: "Next Month", value: addDays(new Date(), 30) },
  { label: "3 Months", value: addDays(new Date(), 90) },
  { label: "6 Months", value: addDays(new Date(), 180) },
];

const formatDate = (date) => {
  if (!date) return "";
  return format(date, "PPP");
};

export function StudyPlanner({ onSave }) {
  const [testDate, setTestDate] = useState();
  const [chapters, setChapters] = useState([
    { id: "1", name: "Introduction", timeSpent: 0, completed: false },
    { id: "2", name: "Basic Concepts", timeSpent: 0, completed: false },
    { id: "3", name: "Advanced Topics", timeSpent: 0, completed: false },
    { id: "4", name: "Case Studies", timeSpent: 0, completed: false },
    { id: "5", name: "Review", timeSpent: 0, completed: false },
  ]);
  const [newChapterName, setNewChapterName] = useState("");
  const [files, setFiles] = useState([]);
  const fileInputRef = useRef(null);

  const completedChapters = chapters.filter(chapter => chapter.completed);
  const remainingChapters = chapters.filter(chapter => !chapter.completed);
  const progress = chapters.length > 0 ? (completedChapters.length / chapters.length) * 100 : 0;

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      setFiles(prev => [...prev, ...newFiles]);
    }
  };

  const handleRemoveFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleAddChapter = () => {
    if (newChapterName.trim()) {
      setChapters(prev => [
        ...prev,
        {
          id: Date.now().toString(),
          name: newChapterName,
          timeSpent: 0,
          completed: false,
        },
      ]);
      setNewChapterName("");
    }
  };

  const handleToggleComplete = (id) => {
    setChapters(prev =>
      prev.map(chapter =>
        chapter.id === id
          ? { ...chapter, completed: !chapter.completed }
          : chapter
      )
    );
  };

  const handleUpdateTime = (id, time) => {
    setChapters(prev =>
      prev.map(chapter =>
        chapter.id === id
          ? { ...chapter, timeSpent: time }
          : chapter
      )
    );
  };

  const handleSave = () => {
    if (onSave) {
      onSave({
        testDate,
        chapters,
        files,
      });
    }
  };

  return (
    <Card className="w-full max-w-4xl mx-auto bg-white/95 backdrop-blur-sm shadow-xl border-gray-100">
      <CardHeader className="space-y-1">
        <CardTitle className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
          Study Planner
        </CardTitle>
        <CardDescription className="text-gray-500">
          Track your progress, manage study materials, and prepare for your test
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-4">
          <div className="flex flex-col space-y-2">
            <Label htmlFor="test-date" className="text-sm font-medium text-gray-700">Test Date</Label>
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "w-full md:w-[300px] justify-start text-left font-normal border-gray-200 hover:bg-gray-50",
                    !testDate && "text-gray-500"
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4 text-indigo-500" />
                  {testDate ? formatDate(testDate) : "Select test date"}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <div className="p-4 border-b border-gray-100">
                  <div className="space-y-2">
                    {presets.map((preset) => (
                      <Button
                        key={preset.label}
                        variant="ghost"
                        className="w-full justify-start font-normal hover:bg-gray-50"
                        onClick={() => setTestDate(preset.value)}
                      >
                        {preset.label}
                      </Button>
                    ))}
                  </div>
                </div>
                <Calendar
                  value={testDate}
                  onChange={setTestDate}
                  className="rounded-md border-0"
                />
              </PopoverContent>
            </Popover>
          </div>
          
          <div className="mt-6">
            <div className="flex items-center justify-between mb-2">
              <Label className="text-sm font-medium text-gray-700">Overall Progress</Label>
              <span className="text-sm text-gray-500">
                {completedChapters.length} of {chapters.length} chapters completed
              </span>
            </div>
            <Progress value={progress} className="h-2 bg-gray-100" />
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label className="text-sm font-medium text-gray-700">Study Materials</Label>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 border-gray-200 hover:bg-gray-50"
            >
              <FileUp className="h-4 w-4 text-indigo-500" />
              Upload Files
            </Button>
            <Input
              type="file"
              ref={fileInputRef}
              className="hidden"
              onChange={handleFileChange}
              multiple
            />
          </div>
          
          {files.length > 0 ? (
            <div className="space-y-3 mt-2">
              {files.map((file, index) => (
                <div 
                  key={index} 
                  className="group flex items-center justify-between p-3 rounded-xl border border-gray-100 bg-white shadow-sm hover:shadow-md transition-all duration-200"
                >
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-indigo-50 text-indigo-500">
                      <Upload className="h-4 w-4" />
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-gray-900 truncate max-w-[300px]">
                        {file.name}
                      </span>
                      <span className="text-xs text-gray-500">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </span>
                    </div>
                  </div>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={() => handleRemoveFile(index)}
                    className="h-8 w-8 p-0 hover:bg-red-50 hover:text-red-600"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          ) : (
            <div 
              onClick={() => fileInputRef.current?.click()}
              className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50/50 cursor-pointer hover:bg-gray-50 transition-colors"
            >
              <div className="p-3 rounded-full bg-indigo-50 text-indigo-600 mb-4">
                <Upload className="h-6 w-6" />
              </div>
              <p className="text-sm font-medium text-gray-900">Drop your files here or click to upload</p>
              <p className="text-xs text-gray-500 mt-1">PDF, DOC, DOCX up to 10MB each</p>
            </div>
          )}
        </div>

        <Tabs defaultValue="completed" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="completed" className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4" />
              Completed Chapters
            </TabsTrigger>
            <TabsTrigger value="remaining" className="flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              Remaining Chapters
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="completed" className="space-y-4 mt-4">
            {completedChapters.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-gray-500 bg-gray-50/50 rounded-lg border-2 border-dashed">
                <CheckCircle className="h-12 w-12 mb-4 text-gray-400" />
                <p className="text-sm">No completed chapters yet</p>
              </div>
            ) : (
              <div className="space-y-3">
                {completedChapters.map(chapter => (
                  <div 
                    key={chapter.id} 
                    className="group flex items-center justify-between p-4 rounded-xl border border-gray-100 bg-white shadow-sm hover:shadow-md transition-all duration-200"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-indigo-50 text-indigo-600 group-hover:bg-indigo-100 transition-colors">
                        <BookMarked className="h-5 w-5" />
                      </div>
                      <span className="font-medium text-gray-900">{chapter.name}</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2 bg-gray-50 px-3 py-1 rounded-lg">
                        <Clock className="h-4 w-4 text-gray-500" />
                        <Input
                          type="number"
                          value={chapter.timeSpent || 0}
                          onChange={(e) => handleUpdateTime(chapter.id, parseInt(e.target.value) || 0)}
                          className="w-16 h-8 text-sm border-0 bg-transparent focus:ring-0"
                        />
                        <span className="text-sm text-gray-500">min</span>
                      </div>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        onClick={() => handleToggleComplete(chapter.id)}
                        className="hover:bg-gray-100 text-gray-700"
                      >
                        Mark Incomplete
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="remaining" className="space-y-4 mt-4">
            {remainingChapters.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-gray-500 bg-gray-50/50 rounded-lg border-2 border-dashed">
                <BookOpen className="h-12 w-12 mb-4 text-gray-400" />
                <p className="text-sm">All chapters completed!</p>
              </div>
            ) : (
              <div className="space-y-3">
                {remainingChapters.map(chapter => (
                  <div 
                    key={chapter.id} 
                    className="group flex items-center justify-between p-4 rounded-xl border border-gray-100 bg-white shadow-sm hover:shadow-md transition-all duration-200"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-gray-50 text-gray-500 group-hover:bg-gray-100 transition-colors">
                        <BookOpen className="h-5 w-5" />
                      </div>
                      <span className="font-medium text-gray-900">{chapter.name}</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2 bg-gray-50 px-3 py-1 rounded-lg">
                        <Clock className="h-4 w-4 text-gray-500" />
                        <Input
                          type="number"
                          value={chapter.timeSpent || 0}
                          onChange={(e) => handleUpdateTime(chapter.id, parseInt(e.target.value) || 0)}
                          className="w-16 h-8 text-sm border-0 bg-transparent focus:ring-0"
                        />
                        <span className="text-sm text-gray-500">min</span>
                      </div>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        onClick={() => handleToggleComplete(chapter.id)}
                        className="hover:bg-indigo-50 text-indigo-600"
                      >
                        Mark Complete
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
            
            <div className="flex items-center gap-2 mt-4">
              <Input
                placeholder="Add new chapter"
                value={newChapterName}
                onChange={(e) => setNewChapterName(e.target.value)}
                className="flex-1"
              />
              <Button onClick={handleAddChapter}>Add</Button>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter className="flex justify-end space-x-2 bg-gray-50/50">
        <Button
          variant="outline"
          onClick={() => window.location.reload()}
          className="border-gray-200 hover:bg-gray-50"
        >
          Reset
        </Button>
        <Button
          onClick={handleSave}
          className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-700 hover:to-purple-700"
        >
          Save Study Plan
        </Button>
      </CardFooter>
    </Card>
  );
}
