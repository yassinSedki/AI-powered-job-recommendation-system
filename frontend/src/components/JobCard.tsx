import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { MapPin, Briefcase, GraduationCap, Clock } from 'lucide-react';
import { JobRecommendation } from '@/lib/api';
import { useNavigate } from 'react-router-dom';

interface JobCardProps {
  job: JobRecommendation;
  onViewOnMap?: () => void;
}

export default function JobCard({ job, onViewOnMap }: JobCardProps) {
  const navigate = useNavigate();

  return (
    <Card className="p-6 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 bg-card">
      <div className="space-y-4">
        <div>
          <h3 className="text-xl font-bold text-foreground mb-2">{job.Role}</h3>
          <p className="text-lg text-muted-foreground font-medium">{job.Company}</p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <MapPin className="w-4 h-4" />
            <span>{job.location}</span>
          </div>
          
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Briefcase className="w-4 h-4" />
            <span>{job.Work_Type}</span>
          </div>

          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <GraduationCap className="w-4 h-4" />
            <span>{job.Qualifications}</span>
          </div>

          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Clock className="w-4 h-4" />
            <span>{job.Experience}</span>
          </div>
        </div>

        {job.salary_range && (
          <div className="pt-2">
            <Badge variant="secondary" className="text-sm">
              {job.salary_range}
            </Badge>
          </div>
        )}

        <div className="flex gap-2 pt-4">
          <Button
            onClick={() => navigate(`/job/${job.Job_Id ?? job.id}`)}
            className="flex-1 bg-primary hover:bg-primary/90"
          >
            View Details
          </Button>
          {onViewOnMap && (
            <Button
              onClick={onViewOnMap}
              variant="outline"
              className="flex-1"
            >
              View on Map
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}
