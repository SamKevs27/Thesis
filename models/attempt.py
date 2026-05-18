import json
from datetime import datetime
from .database import db


class Attempt(db.Model):
    __tablename__ = 'attempts'
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    movement_type = db.Column(db.String(20), nullable=False, index=True)
    overall_score = db.Column(db.Float, nullable=False)
    wiraga_score  = db.Column(db.Float, nullable=False)
    wirama_score  = db.Column(db.Float, nullable=False)
    video_path    = db.Column(db.String(255))
    composed_path = db.Column(db.String(255))
    feedback_json = db.Column(db.Text)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            'id':            self.id,
            'movement_type': self.movement_type,
            'overall_score': self.overall_score,
            'wiraga_score':  self.wiraga_score,
            'wirama_score':  self.wirama_score,
            'video_path':    self.video_path,
            'composed_path': self.composed_path,
            'feedback':      json.loads(self.feedback_json) if self.feedback_json else [],
            'created_at':    self.created_at.isoformat(),
        }
