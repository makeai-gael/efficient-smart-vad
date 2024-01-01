# efficient-smart-vad
A very low performance wake up word detector that combine old fashion audio processing with machine learning to train and run your own custom wake up word.
NB: Repo in development

# installation
# Requirement
- python >=3.8
- for full code: `pip3 install  -r requirements.txt`
- for inference only: TBD

# Audio data processing
## Audio folder
- download the ambient sounds from this link and place all sounds in `data/dataset/ambient-sounds/` or add your own ambient sounds if need be.
- find and generate your custom wake-word sounds and place them in `data/dataset/wake-word/`

## Audio format
the following format are supported: '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.aiff', '.ra', '.mid' & '.midi'

## Wake-word requirements
- aproximately 1 second in length
- Any sampling rate from 8khz to 24khz (all sounds will be reduced to 8khz)
- Insure the the wake-word sound is as clear and clean as possible
- You must have at least 30 clear and diverse sound of your wake-word for good results. (the more the better)
- You must inlcude different accent and pronouciations to use this for public use.
