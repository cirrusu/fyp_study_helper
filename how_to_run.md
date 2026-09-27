1. Create a virtual environment in the code folder

   1. code: py -m venv .venv
2. Install required packages

   1. code: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
3. Start application

   1. code: .\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
4. Open browser and go to the enter the following:

   1. http://127.0.0.1:8000/health - this shows the 
   2. http://127.0.0.1:8000/docs - this is the applications interactive API page
5. Open a new powershell terminal in the same folder
6. Try automatic demonstration:

   1. code: .\\.venv\\Scripts\\python.exe examples/demo\_client.py







Testing the API

1. Manually create a student.

   1. Keep server open and open http://127.0.0.1:8000/docs
   2. Create a profile with POST /v1/students

      1. code: 
{
    "selected\_topics": \["loops", "functions"],
    "preferred\_format": "exercise",
    "target\_level": "beginner"
}
      2. Expect HTTP 201 - means student profile was created
      3. Authorise the student. Copy the token number, click Authorize button on the top, and paste the token number before pressing authorize again
2. Inspect profile with GET /v1/me
3. Read questions with GET /v1/quiz/questions

   1. Enter loops for topic and read the questions
4. Submit answers with POST /v1/quiz/attempts

   1. code:
{
    "submission\_id": "manual-quiz-001",
    "answers": \[
                {"question\_id": "loops-1", "selected\_option": "b"},
                {"question\_id": "loops-2", "selected\_option": "a"}]
}
   2. The two answers are deliberately wrong
5. Request recommendations and test feedback from POST /v1/recommendations

   1. code:{
         "topics": \["loops"],
         "k": 5,
         "max\_minutes": 15,
         "algorithm": "performance"}
6. Give feedback at POST /v1/feedback

   1. code: {
          "resource\_id": "exercise-loop-sum",
          "helpful": true,
          "completed": true}



