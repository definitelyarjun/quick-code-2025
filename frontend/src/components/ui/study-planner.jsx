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

const formatDate = (date) => {
  if (!date) return "";
  const options = { year: 'numeric', month: 'long', day: 'numeric' };
  return date.toLocaleDateString('en-US', options);
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
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Study Planner</CardTitle>
        <CardDescription>
          Track your progress, manage study materials, and prepare for your test
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="test-date">Test Date</Label>
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "w-[240px] justify-start text-left font-normal",
                    !testDate && "text-muted-foreground"
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {testDate ? formatDate(testDate) : "Select test date"}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  value={testDate}
                  onChange={setTestDate}
                />
              </PopoverContent>
            </Popover>
          </div>
          
          <div className="mt-6">
            <div className="flex items-center justify-between mb-2">
              <Label>Overall Progress</Label>
              <span className="text-sm text-muted-foreground">
                {completedChapters.length} of {chapters.length} chapters completed
              </span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label>Study Materials</Label>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2"
            >
              <FileUp className="h-4 w-4" />
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
          
          {files.length > 0 && (
            <div className="space-y-2 mt-2">
              {files.map((file, index) => (
                <div 
                  key={index} 
                  className="flex items-center justify-between p-2 rounded-md border bg-background"
                >
                  <div className="flex items-center gap-2">
                    <Upload className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm truncate max-w-[300px]">{file.name}</span>
                  </div>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={() => handleRemoveFile(index)}
                    className="h-8 w-8 p-0"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ))}
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
              <div className="text-center py-8 text-muted-foreground">
                No completed chapters yet
              </div>
            ) : (
              <div className="space-y-2">
                {completedChapters.map(chapter => (
                  <div 
                    key={chapter.id} 
                    className="flex items-center justify-between p-3 rounded-md border bg-muted/50"
                  >
                    <div className="flex items-center gap-2">
                      <BookMarked className="h-4 w-4 text-primary" />
                      <span>{chapter.name}</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-1">
                        <Clock className="h-4 w-4 text-muted-foreground" />
                        <Input
                          type="number"
                          value={chapter.timeSpent || 0}
                          onChange={(e) => handleUpdateTime(chapter.id, parseInt(e.target.value) || 0)}
                          className="w-16 h-8 text-sm"
                        />
                        <span className="text-sm text-muted-foreground">min</span>
                      </div>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        onClick={() => handleToggleComplete(chapter.id)}
                        className="h-8"
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
              <div className="text-center py-8 text-muted-foreground">
                All chapters completed!
              </div>
            ) : (
              <div className="space-y-2">
                {remainingChapters.map(chapter => (
                  <div 
                    key={chapter.id} 
                    className="flex items-center justify-between p-3 rounded-md border"
                  >
                    <div className="flex items-center gap-2">
                      <BookOpen className="h-4 w-4 text-muted-foreground" />
                      <span>{chapter.name}</span>
                    </div>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => handleToggleComplete(chapter.id)}
                      className="h-8"
                    >
                      Mark Complete
                    </Button>
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
      <CardFooter className="flex justify-between">
        <Button variant="outline">Reset</Button>
        <Button onClick={handleSave}>Save Study Plan</Button>
      </CardFooter>
    </Card>
  );
}
