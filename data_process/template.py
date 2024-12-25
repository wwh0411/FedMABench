template_describe_picture = """A user is performing a *task* on a mobile phone, going through several steps to complete it. 
Each step involves a previous "User Action" that leads to the interface shown in the screenshot. 

The screenshot provided captures the interface of current step during the *task*.
Please describe the screenshot briefly. 

## Answer Format
Keep your response concise and only output important things. Try to capture the main information. 

## Your Answer
The screenshot shows: 
"""

old  = """A user is performing a *task* on a mobile phone, going through several steps to complete it. 
Each step involves a previous "User Action" that leads to the interface shown in the screenshot. 
Each step involves a previous "User Action" that leads to the interface shown in the screenshot. 

The screenshot provided captures the interface of current step during the *task*.
Based on the user's previous action and the current screenshot, describe what the mobile user is trying to do in this step. """


template_describe_action = """A user is performing a *task* on a mobile phone, progressing through multiple steps to complete the task. 
Each step involves an interface shown in the provided screenshot, and a "User Action" performed to move on to the next step.

Based on the current screenshot and the user’s action, describe the specific goal the user is trying to achieve in this step of the *task*.

## User Action
{act}

## Answer Format
Keep your response concise and capture the important things, focusing on key details like the app name, email address, search terms, item name, and title.

## Your Answer
The user is trying to: 
"""

template_final = {
    # template without image
    "without_image": """A user is performing a *task* on a mobile phone, progressing through multiple steps to complete the task. 
Each step involves an interface shown in the provided screenshot, and a "User Action" performed to move on to the next step.
The "User Action" is provided in the context *Process*.
The *task* is not known. Now based on the *Process*, describe the mobile user's *task* when completing these actions.

## Process
{history}

## Answer Format
Keep your answer concise and clear, as if the user were explaining the *task* to someone else in one sentence.
Include key details like the app name, individual name, email address, search terms, item name, and title.

## Your Answer
The user is trying to:
""",
    # template with image
    "with_image": """A user is performing a *task* on a mobile phone, progressing through multiple steps to complete the task. 
Each step involves an interface shown in the provided screenshot, and a "User Action" performed to move on to the next step.
The "User Action" is provided in the context *Process*.
I am also providing a picture that shows all of the screenshots concatenated together.
The *task* is not known. Now based on the *Process* and the picture, describe the mobile user's *task* when completing these actions.

## Process
{history}

## Answer Format
Keep your answer concise and clear, as if the user were explaining the *task* to someone else in one sentence.

## Your Answer
The task is: 
"""
}


template_final_w_des = {
    # template without image
    "without_image": """A user is performing a *task* on a mobile phone, progressing through multiple steps to complete the task. 
Each step involves an interface shown in the provided screenshot, and a "User Action" performed to move on to the next step.
The "User Action" and "Screenshot Description" are provided in the context *Process*.
The *task* is not known. Now based on the *Process*, describe the mobile user's *task* when completing these actions.

## Process
{history}

## Answer Format
Keep your answer concise and clear, as if the user were explaining the *task* to someone else in one sentence.
Include key details like the app name, individual name, email address, search terms, item name, and title.

## Your Answer
The user is trying to:
""",
    # template with image
    "with_image": """A user is performing a *task* on a mobile phone, progressing through multiple steps to complete the task. 
Each step involves an interface shown in the provided screenshot, and a "User Action" performed to move on to the next step.
The "User Action" and "Screenshot Description" are provided in the context *Process*.
I am also providing a picture that shows all of the screenshots concatenated together.
The *task* is not known. Now based on the *Process* and the picture, describe the mobile user's *task* when completing these actions.

## Process
{history}

## Answer Format
Keep your answer concise and clear, as if the user were explaining the *task* to someone else in one sentence.

## Your Answer
The task is: 
"""
}


template_6_sub = """### Step {i}
### Screenshot Description
{pic_des}
### User Action
{act_des}

"""

template_4_sub = """### Step {i}
### User Action
{act_des}

"""

template_1_wo_action = """I am providing a picture that shows several concatenated screenshots. 
These screenshots capture a *user* performing a *task* on a mobile phone. 
Please analyze the images and infer what the *task* is. 
Your answer should be concise, clear, and directly output the *task*.
Answer in a tone that you are telling this *task* to the *user*.
"""

template_2_w_action = """A *user* performing a *task* on a mobile phone.
I am providing the descriptions of the actions a user does during the process of the task.
Please infer what the *task* is. 

## Descriptions
{descriptions}

## Answer format
Your answer should be concise, clear, and directly output the *task*.
Answer in a tone that you are telling this *task* to the *user*.

## Your Answer
The task is: 
"""


template_train_hl = """<image>You are a smartphone assistant tasked with helping users complete actions by interacting with apps.
I will provide you with one screenshot, representing the UI state before an operation is performed. 

For the screenshot, you need to identify and output a specific action required to complete the **User Instruction**

### User Instruction ###
{ins}

### Response Requirements ###
For each screenshot, you need to decide just one action on the current screenshot.
You must choose one of the actions below:
1. **Click on button with the text "<UI element text>"**  
If the button has no text related, just output "Click on button".

2. **Long press on button with the text "<UI element text>"**  

3. **Type text: "<input text>"**  
Type the <input text> in the current input field or search bar.

4. **Scroll <direction>**  
Scroll the UI element by <direction>.  
If the current UI includes scrollers but lacks the necessary elements for the task, try scrolling down to reveal elements below or scrolling up to uncover elements above. 
Similarly, scroll right to reveal elements on the right or scroll left to uncover elements on the left.

5. **Return to the home page**  
Return to the home page. If you want to exit an app, use this action.

6. **Go back to the previous page**  
Go back to the previous page. If you need to return to the previous step or undo an action, use this action to navigate back.

7. **Open App: <app name>**  
If you wish to open an app, use this action to open <app name>.

8. **Wait for response**  
Pause for a moment to allow any background processes to complete or for elements to load before proceeding with the next action.

9. **Check status: <successful/infeasible>**  
If you think all the requirements of the user's instruction have been completed successfully and no further operation is required, you can choose "successful" to terminate the operation process.  
If the task cannot be completed due to missing elements or any other issue, you can use "infeasible" to indicate that the action cannot be performed.

### Your Response ###

"""


template_train_ll = """<image>You are a smartphone assistant tasked with helping users complete actions by interacting with apps.
I will provide you with one screenshot, representing the UI state before an operation is performed. 

For the screenshot, you need to identify and output a specific action required to complete the **User Instruction**.
The subordinate instruction for the current step is also provided as the **Subordinate Instruction**. 

### User Instruction ###
{ins}

### Subordinate Instruction ###
{sub_ins}

### Response Requirements ###
For each screenshot, you need to decide just one action on the current screenshot.
You must choose one of the actions below:
1. **Click on button with the text "<UI element text>"**  
If the button has no text related, just output "Click on button".

2. **Long press on button with the text "<UI element text>"**  

3. **Type text: "<input text>"**  
Type the <input text> in the current input field or search bar.

4. **Scroll <direction>**  
Scroll the UI element by <direction>.  
If the current UI includes scrollers but lacks the necessary elements for the task, try scrolling down to reveal elements below or scrolling up to uncover elements above. 
Similarly, scroll right to reveal elements on the right or scroll left to uncover elements on the left.

5. **Return to the home page**  
Return to the home page. If you want to exit an app, use this action.

6. **Go back to the previous page**  
Go back to the previous page. If you need to return to the previous step or undo an action, use this action to navigate back.

7. **Open App: <app name>**  
If you wish to open an app, use this action to open <app name>.

8. **Wait for response**  
Pause for a moment to allow any background processes to complete or for elements to load before proceeding with the next action.

9. **Check status: <successful/infeasible>**  
If you think all the requirements of the user's instruction have been completed successfully and no further operation is required, you can choose "successful" to terminate the operation process.  
If the task cannot be completed due to missing elements or any other issue, you can use "infeasible" to indicate that the action cannot be performed.

### Your Response ###

"""