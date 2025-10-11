import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import MapPicker from '@/components/MapPicker';
import { apiClient, JobRecommendation } from '@/lib/api';
import { Loader2, MapPin, Briefcase, GraduationCap, Clock, DollarSign, ArrowLeft, FileText, ListChecks, Building2, Globe, Heart } from 'lucide-react';
import { toast } from 'sonner';

export default function JobDetailsPage() {
  const { id } = useParams<{ id: string }>();
  const [job, setJob] = useState<JobRecommendation | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchJob = async () => {
      if (!id) return;
      
      setLoading(true);
      try {
        const jobData = await apiClient.getJob(id);
        setJob(jobData);
      } catch (error) {
        console.error('Error fetching job:', error);
        toast.error('Failed to load job details');
      } finally {
        setLoading(false);
      }
    };

    fetchJob();
  }, [id]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!job) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-2">Job Not Found</h2>
          <p className="text-muted-foreground mb-4">The job you're looking for doesn't exist.</p>
          <Button asChild>
            <Link to="/recommend">Back to Recommendations</Link>
          </Button>
        </div>
      </div>
    );
  }

  const skills = typeof job['skills'] === 'string'
    ? job['skills']
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
    : [];

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="container max-w-5xl">
        <Button variant="ghost" asChild className="mb-6">
          <Link to="/recommend">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Recommendations
          </Link>
        </Button>

        <div className="space-y-6">
          {/* Header Card */}
          <Card className="p-8">
            <div className="space-y-4">
              <div>
                <h1 className="text-4xl font-bold mb-2">{job['Job Title'] ?? job.Role}</h1>
                <p className="text-2xl text-muted-foreground font-medium">{job.Company}</p>
              </div>

              {job.salary_range && (
                <div className="inline-flex items-center gap-2 bg-gradient-to-r from-secondary to-secondary/80 text-secondary-foreground px-4 py-2 rounded-lg">
                  <DollarSign className="w-5 h-5" />
                  <span className="font-semibold">{job.salary_range}</span>
                </div>
              )}
            </div>
          </Card>

          {/* Modern Details Grid */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Job Details */}
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4">Job Details</h2>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <Briefcase className="w-5 h-5 text-muted-foreground mt-0.5" />
                  <div>
                    <p className="font-medium">Work Type</p>
                    <p className="text-muted-foreground">{job.Work_Type}</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <Building2 className="w-5 h-5 text-muted-foreground mt-0.5" />
                  <div>
                    <p className="font-medium">Company Size</p>
                    <p className="text-muted-foreground">{job['Company Size']}</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <Clock className="w-5 h-5 text-muted-foreground mt-0.5" />
                  <div>
                    <p className="font-medium">Experience</p>
                    <p className="text-muted-foreground">{job.Experience}</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <DollarSign className="w-5 h-5 text-muted-foreground mt-0.5" />
                  <div>
                    <p className="font-medium">Salary Range</p>
                    <p className="text-muted-foreground">{job.salary_range ?? job['Salary Range']}</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <Globe className="w-5 h-5 text-muted-foreground mt-0.5" />
                  <div>
                    <p className="font-medium">Country</p>
                    <p className="text-muted-foreground">{job['Country']}</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <MapPin className="w-5 h-5 text-muted-foreground mt-0.5" />
                  <div>
                    <p className="font-medium">Location</p>
                    <p className="text-muted-foreground">{job.location}</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <GraduationCap className="w-5 h-5 text-muted-foreground mt-0.5" />
                  <div>
                    <p className="font-medium">Qualification</p>
                    <p className="text-muted-foreground">{job.Qualifications}</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <ListChecks className="w-5 h-5 text-muted-foreground mt-0.5" />
                  <div>
                    <p className="font-medium">Preference</p>
                    <p className="text-muted-foreground">{job['Preference']}</p>
                  </div>
                </div>
              </div>
            </Card>

            {/* Description & Benefits */}
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4">Description & Benefits</h2>
              <div className="space-y-6">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <FileText className="w-5 h-5 text-muted-foreground" />
                    <p className="font-medium">Job Description</p>
                  </div>
                  <p className="text-muted-foreground leading-relaxed">
                    {job['Job Description'] ?? 'No description provided.'}
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Heart className="w-5 h-5 text-muted-foreground" />
                    <p className="font-medium">Benefits</p>
                  </div>
                  <p className="text-muted-foreground leading-relaxed">
                    {job['Benefits'] ?? 'No benefits listed.'}
                  </p>
                </div>
              </div>
            </Card>
          </div>

          {/* Skills & Responsibilities */}
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Skills & Responsibilities</h2>
            <div className="space-y-6">
              <div className="space-y-3">
                <p className="font-medium">Skills</p>
                {skills.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {skills.map((skill) => (
                      <Badge key={skill} variant="secondary" className="px-3 py-1 rounded-full">
                        {skill}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <p className="text-muted-foreground">No skills provided.</p>
                )}
              </div>

              <div className="space-y-3">
                <p className="font-medium">Responsibilities</p>
                <p className="text-muted-foreground leading-relaxed">
                  {job['Responsibilities'] ?? 'No responsibilities provided.'}
                </p>
              </div>
            </div>
          </Card>

          {/* Map */}
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Job Location</h2>
            <MapPicker
              latitude={job.latitude}
              longitude={job.longitude}
              onLocationSelect={() => {}}
              markers={[
                {
                  id: job.Job_Id ?? job.id,
                  lat: job.latitude,
                  lng: job.longitude,
                  title: job.Role,
                  company: job.Company,
                },
              ]}
              zoom={12}
            />
          </Card>

          {/* Action Button */}
          <div className="flex justify-center pt-4">
            <Button size="lg" className="bg-gradient-to-r from-primary to-accent">
              Apply Now
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
