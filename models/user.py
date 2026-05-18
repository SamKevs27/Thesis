from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from .database import db


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name     = db.Column(db.String(100), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    last_login    = db.Column(db.DateTime)

    attempts = db.relationship('Attempt', backref='user', lazy='dynamic',
                               cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_best_score(self, movement_type):
        from models.attempt import Attempt
        best = self.attempts.filter_by(movement_type=movement_type)\
                            .order_by(Attempt.overall_score.desc()).first()
        return best.overall_score if best else None

    def get_attempt_count(self, movement_type=None):
        q = self.attempts
        if movement_type:
            q = q.filter_by(movement_type=movement_type)
        return q.count()
