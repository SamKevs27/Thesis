from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from models.user import User


class LoginForm(FlaskForm):
    email    = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember me')


class RegisterForm(FlaskForm):
    full_name = StringField('Full Name', validators=[DataRequired(), Length(min=2, max=100)])
    email     = StringField('Email', validators=[DataRequired(), Email()])
    password  = PasswordField('Password',
                              validators=[DataRequired(), Length(min=8, max=128)])
    confirm   = PasswordField('Confirm Password',
                              validators=[DataRequired(), EqualTo('password',
                                          message='Passwords must match')])

    def validate_email(self, field):
        if User.query.filter_by(email=field.data.lower().strip()).first():
            raise ValidationError('Email already registered')
