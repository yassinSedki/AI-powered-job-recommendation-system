import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Card } from '@/components/ui/card';
import MapPicker from '@/components/MapPicker';
import { apiClient, PredictSalaryRequest } from '@/lib/api';
import { toast } from 'sonner';
import { Loader2, DollarSign, TrendingUp } from 'lucide-react';

export default function SalaryPredictPage() {
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    role: '',
    skills: '',
    qualification: '',
    work_type: '',
    experience_mid: 0,
    latitude: 36.8065,
    longitude: 10.1815,
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [predictedSalary, setPredictedSalary] = useState<number | null>(null);

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.role.trim()) newErrors.role = 'Role is required';
    if (!formData.skills.trim()) newErrors.skills = 'Skills are required';
    if (!formData.qualification) newErrors.qualification = 'Education is required';
    if (!formData.work_type) newErrors.work_type = 'Work type is required';
    if (formData.experience_mid < 0) newErrors.experience_mid = 'Experience must be positive';
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      toast.error('Please fill in all required fields');
      return;
    }

    setLoading(true);
    try {
      const request: PredictSalaryRequest = {
        role: formData.role,
        skills: formData.skills,
        work_type: formData.work_type,
        qualification: formData.qualification,
        experience_mid: formData.experience_mid,
        latitude: formData.latitude,
        longitude: formData.longitude,
      };

      const response = await apiClient.predictSalary(request);
      setPredictedSalary(response.predicted_salary);
      toast.success('Salary prediction complete!');
      
      // Scroll to results
      setTimeout(() => {
        document.getElementById('salary-result')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      role: '',
      skills: '',
      qualification: '',
      work_type: '',
      experience_mid: 0,
      latitude: 36.8065,
      longitude: 10.1815,
    });
    setPredictedSalary(null);
    setErrors({});
  };

  const handleLocationSelect = (lat: number, lng: number) => {
    setFormData({ ...formData, latitude: lat, longitude: lng });
    toast.success('Location updated');
  };

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="container max-w-7xl">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Salary Prediction</h1>
          <p className="text-muted-foreground text-lg">
            Get an accurate salary estimate based on your skills and location
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Form Section */}
          <Card className="p-6">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="role">Role *</Label>
                <Input
                  id="role"
                  value={formData.role}
                  onChange={(e) => setFormData({ ...formData, role: e.target.value })}
                  placeholder="e.g., Software Engineer"
                  className={errors.role ? 'border-destructive' : ''}
                />
                {errors.role && <p className="text-sm text-destructive">{errors.role}</p>}
              </div>

              <div className="space-y-2">
                <Label htmlFor="skills">Skills *</Label>
                <Textarea
                  id="skills"
                  value={formData.skills}
                  onChange={(e) => setFormData({ ...formData, skills: e.target.value })}
                  placeholder="e.g., React, TypeScript, Node.js"
                  className={errors.skills ? 'border-destructive' : ''}
                  rows={3}
                />
                {errors.skills && <p className="text-sm text-destructive">{errors.skills}</p>}
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="qualification">Education *</Label>
                  <Select
                    value={formData.qualification}
                    onValueChange={(value) => setFormData({ ...formData, qualification: value })}
                  >
                    <SelectTrigger className={errors.qualification ? 'border-destructive' : ''}>
                      <SelectValue placeholder="Select education" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Bachelor">Bachelor</SelectItem>
                      <SelectItem value="Master">Master</SelectItem>
                      <SelectItem value="PhD">PhD</SelectItem>
                    </SelectContent>
                  </Select>
                  {errors.qualification && <p className="text-sm text-destructive">{errors.qualification}</p>}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="work_type">Work Type *</Label>
                  <Select
                    value={formData.work_type}
                    onValueChange={(value) => setFormData({ ...formData, work_type: value })}
                  >
                    <SelectTrigger className={errors.work_type ? 'border-destructive' : ''}>
                      <SelectValue placeholder="Select type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Contract">Contract</SelectItem>
                      <SelectItem value="Part-Time">Part-Time</SelectItem>
                      <SelectItem value="Intern">Intern</SelectItem>
                      <SelectItem value="Full-Time">Full-Time</SelectItem>
                    </SelectContent>
                  </Select>
                  {errors.work_type && <p className="text-sm text-destructive">{errors.work_type}</p>}
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="experience_mid">Experience (years) *</Label>
                  <Input
                    id="experience_mid"
                    type="number"
                    min="0"
                    value={formData.experience_mid}
                    onChange={(e) => setFormData({ ...formData, experience_mid: parseInt(e.target.value) || 0 })}
                    className={errors.experience_mid ? 'border-destructive' : ''}
                  />
                  {errors.experience_mid && <p className="text-sm text-destructive">{errors.experience_mid}</p>}
                </div>
              </div>


              <div className="flex gap-4">
                <Button
                  type="submit"
                  disabled={loading}
                  className="flex-1 bg-gradient-to-r from-primary to-accent"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Calculating...
                    </>
                  ) : (
                    <>
                      <TrendingUp className="mr-2 h-4 w-4" />
                      Predict Salary
                    </>
                  )}
                </Button>
                <Button type="button" variant="outline" onClick={handleReset}>
                  Reset
                </Button>
              </div>
            </form>
          </Card>

          {/* Map Section */}
          <div className="space-y-4">
            <div>
              <h2 className="text-xl font-semibold mb-2">Select Your Location</h2>
              <p className="text-sm text-muted-foreground mb-4">
                Click on the map to set your coordinates
              </p>
            </div>
            <MapPicker
              latitude={formData.latitude}
              longitude={formData.longitude}
              onLocationSelect={handleLocationSelect}
            />
          </div>
        </div>

        {/* Results Section */}
        {predictedSalary !== null && (
          <div id="salary-result" className="mt-12">
            <Card className="p-8 bg-gradient-to-r from-primary to-accent text-primary-foreground">
              <div className="text-center space-y-4">
                <div className="inline-block bg-white/20 p-4 rounded-full mb-4">
                  <DollarSign className="w-12 h-12" />
                </div>
                <h2 className="text-2xl font-semibold">Your Estimated Salary</h2>
                <p className="text-5xl font-bold">
                  ${predictedSalary.toLocaleString()}
                </p>
                <p className="text-xl opacity-90">per year</p>
                <p className="text-sm opacity-80 max-w-2xl mx-auto pt-4">
                  This salary estimate is based on your role, skills, experience, and location. 
                  Actual salaries may vary depending on the company and specific position.
                </p>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
