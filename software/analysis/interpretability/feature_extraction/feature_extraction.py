import re
import os
import csv
import openai 

# Power: -1 to 1 (greater user power to greater chatbot power)
# Expertise: 0 to 1 (expertise differential)
# Stakes: 0 to 1 (low to high stakes)
# Emotional: 0 to 1 (low to high emotional involvement)
# Formality: 0 to 1 (informal to formal)
# Expected duration: 0 to 1 (shorter to longer term relationships)
scenario_scores = {
    "Chatbot for Customer Support": {
        "power": -0.4,    # Customer has slight power
        "expertise": 0.7,  # Bot has technical knowledge
        "stakes": 0.4,     # Moderate-low stakes
        "emotional": 0.5,  # Can be frustrating
        "formality": 0.6,  # Professional but not rigid
        "duration": 0.1    # One-time interaction
    },
    "Social Media Interaction": {
        "power": 0.0,     # Equal peers
        "expertise": 0.3,  # Varying knowledge levels
        "stakes": 0.3,     # Low-moderate stakes
        "emotional": 0.8,  # Often highly emotional
        "formality": 0.2,  # Very informal
        "duration": 0.1    # Usually one-time
    },
    "Email Exchange in the Workplace": {
        "power": 0.6,     # Manager has authority
        "expertise": 0.4,  # Both have relevant knowledge
        "stakes": 0.7,     # Professional implications
        "emotional": 0.3,  # Professional distance
        "formality": 0.8,  # Formal communication
        "duration": 0.7    # Ongoing relationship
    },
    "Teacher-Student Conversation": {
        "power": 0.8,     # Clear authority
        "expertise": 0.7,  # Teacher knows more
        "stakes": 0.6,     # Educational impact
        "emotional": 0.5,  # Personal but professional
        "formality": 0.7,  # Structured setting
        "duration": 0.8    # Long-term relationship
    },
    "Workplace Conflict Resolution": {
        "power": 0.5,     # Team leader has some authority
        "expertise": 0.4,  # Similar professional level
        "stakes": 0.8,     # High workplace impact
        "emotional": 0.8,  # Very emotional situation
        "formality": 0.7,  # Professional setting
        "duration": 0.7    # Ongoing work relationship
    },
    "Counseling Session (Therapist-Client)": {
        "power": 0.3,     # Collaborative relationship
        "expertise": 0.8,  # Professional expertise
        "stakes": 0.9,     # Personal wellbeing at stake
        "emotional": 0.9,  # Highly emotional
        "formality": 0.6,  # Professional but personal
        "duration": 0.8    # Ongoing relationship
    },
    "Medical Session (Doctor-Patient)": {
        "power": 0.7,     # Medical authority
        "expertise": 0.9,  # High medical expertise
        "stakes": 1.0,     # Health at stake
        "emotional": 0.7,  # Health concerns
        "formality": 0.8,  # Very formal setting
        "duration": 0.5    # Mix of ongoing and one-time
    },
    "Job Interview Simulation": {
        "power": 0.8,     # Clear power dynamic
        "expertise": 0.5,  # Mixed expertise areas
        "stakes": 0.9,     # Career impact
        "emotional": 0.7,  # High stress
        "formality": 0.9,  # Very formal
        "duration": 0.1    # One-time event
    },
    "Mentoring Conversation": {
        "power": 0.4,     # Gentle guidance
        "expertise": 0.7,  # Mentor's experience
        "stakes": 0.6,     # Career development
        "emotional": 0.5,  # Supportive relationship
        "formality": 0.5,  # Semi-formal
        "duration": 0.8    # Long-term relationship
    },
    "Student Seeking Help with Homework": {
        "power": 0.6,     # Teaching dynamic
        "expertise": 0.8,  # Clear knowledge gap
        "stakes": 0.5,     # Academic performance
        "emotional": 0.4,  # Learning focused
        "formality": 0.4,  # Casual teaching
        "duration": 0.3    # Short-term help
    },
    "Customer Requesting a Refund": {
        "power": -0.5,    # Customer leverage
        "expertise": 0.5,  # Policy knowledge
        "stakes": 0.6,     # Financial impact
        "emotional": 0.7,  # Often frustrated
        "formality": 0.7,  # Business transaction
        "duration": 0.1    # One-time interaction
    },
    "Collaborating on a Group Project": {
        "power": 0.0,     # Equal peers
        "expertise": 0.3,  # Similar levels
        "stakes": 0.7,     # Shared outcome
        "emotional": 0.5,  # Team dynamics
        "formality": 0.5,  # Semi-formal
        "duration": 0.6    # Project duration
    },
    "Teacher Helping a Struggling Student": {
        "power": 0.7,     # Teaching authority
        "expertise": 0.8,  # Educational expertise
        "stakes": 0.8,     # Academic success
        "emotional": 0.7,  # Sensitive situation
        "formality": 0.6,  # Supportive setting
        "duration": 0.7    # Ongoing support
    },
    "Employee Asking for a Raise": {
        "power": -0.2,    # Complex power dynamic
        "expertise": 0.4,  # Both have relevant info
        "stakes": 0.9,     # Financial impact
        "emotional": 0.8,  # High stress
        "formality": 0.8,  # Very formal
        "duration": 0.1    # One-time discussion
    },
    "Social Media Disagreement": {
        "power": 0.0,     # Equal footing
        "expertise": 0.2,  # Similar levels
        "stakes": 0.2,     # Low real impact
        "emotional": 0.8,  # Often heated
        "formality": 0.1,  # Very informal
        "duration": 0.1    # One-time
    },
    "Doctor Delivering Test Results": {
        "power": 0.8,     # Medical authority
        "expertise": 0.9,  # Medical expertise
        "stakes": 1.0,     # Health implications
        "emotional": 0.8,  # High anxiety
        "formality": 0.9,  # Very formal
        "duration": 0.4    # Specific event
    },
    "Negotiating a Business Deal": {
        "power": 0.0,     # Equal partners
        "expertise": 0.5,  # Both knowledgeable
        "stakes": 0.9,     # High value
        "emotional": 0.6,  # Professional tension
        "formality": 0.9,  # Very formal
        "duration": 0.5    # Deal-specific
    },
    "Teacher Offering Study Tips": {
        "power": 0.6,     # Teaching authority
        "expertise": 0.7,  # Educational expertise
        "stakes": 0.5,     # Helpful but not crucial
        "emotional": 0.3,  # Practical focus
        "formality": 0.5,  # Semi-formal
        "duration": 0.4    # Specific guidance
    },
    "Handling a Social Media Crisis": {
        "power": -0.6,    # Customer power
        "expertise": 0.6,  # PR expertise
        "stakes": 0.8,     # Reputation risk
        "emotional": 0.9,  # High tension
        "formality": 0.7,  # Professional crisis
        "duration": 0.2    # Crisis-specific
    },
    "Medical Consultation on a Lifestyle Change": {
        "power": 0.6,     # Medical authority
        "expertise": 0.8,  # Health expertise
        "stakes": 0.8,     # Health impact
        "emotional": 0.5,  # Motivational
        "formality": 0.7,  # Professional but supportive
        "duration": 0.7    # Ongoing changes
    },
    "Technical Support Call": {
        "power": -0.3,    # Customer service
        "expertise": 0.8,  # Technical knowledge
        "stakes": 0.5,     # Technical issue
        "emotional": 0.6,  # Can be frustrating
        "formality": 0.6,  # Professional support
        "duration": 0.1    # One-time help
    },
    "Restaurant Reservation Dispute": {
        "power": -0.4,    # Customer service
        "expertise": 0.4,  # Service knowledge
        "stakes": 0.5,     # Dining plans
        "emotional": 0.7,  # Frustration
        "formality": 0.6,  # Service industry
        "duration": 0.1    # One-time
    },
    "Travel Agent Booking a Trip": {
        "power": -0.2,    # Service relationship
        "expertise": 0.7,  # Travel knowledge
        "stakes": 0.7,     # Vacation planning
        "emotional": 0.4,  # Exciting but professional
        "formality": 0.6,  # Professional service
        "duration": 0.3    # Trip planning period
    },
    "Financial Advisor Consultation": {
        "power": 0.4,     # Expert guidance
        "expertise": 0.9,  # Financial expertise
        "stakes": 0.9,     # Financial impact
        "emotional": 0.6,  # Money concerns
        "formality": 0.8,  # Professional finance
        "duration": 0.7    # Ongoing advice
    },
    "Parent-Teacher Conference": {
        "power": 0.3,     # Shared authority
        "expertise": 0.6,  # Educational insight
        "stakes": 0.8,     # Child's education
        "emotional": 0.7,  # Personal investment
        "formality": 0.7,  # Professional but personal
        "duration": 0.6    # School year
    },
    "Real Estate Agent Showing a Property": {
        "power": -0.3,    # Client-driven
        "expertise": 0.7,  # Property knowledge
        "stakes": 0.8,     # Major purchase
        "emotional": 0.6,  # Important decision
        "formality": 0.7,  # Professional sales
        "duration": 0.3    # House hunting period
    },
    "Car Salesperson Negotiation": {
        "power": -0.4,    # Customer leverage
        "expertise": 0.6,  # Product knowledge
        "stakes": 0.8,     # Major purchase
        "emotional": 0.7,  # High pressure
        "formality": 0.6,  # Sales environment
        "duration": 0.2    # Purchase period
    },
    "Library Assistant Helping with Research": {
        "power": 0.3,     # Helpful guide
        "expertise": 0.8,  # Research knowledge
        "stakes": 0.4,     # Academic help
        "emotional": 0.2,  # Information focused
        "formality": 0.5,  # Professional but casual
        "duration": 0.2    # Specific help
    },
    "Gym Trainer Initial Consultation": {
        "power": 0.4,     # Fitness expertise
        "expertise": 0.7,  # Training knowledge
        "stakes": 0.6,     # Health goals
        "emotional": 0.4,  # Motivational
        "formality": 0.5,  # Professional but casual
        "duration": 0.7    # Training relationship
    },
    "Wedding Planner Consultation": {
        "power": -0.2,    # Client-driven
        "expertise": 0.8,  # Event expertise
        "stakes": 0.9,     # Important event
        "emotional": 0.8,  # Very personal
        "formality": 0.7,  # Professional service
        "duration": 0.6    # Planning period
    },
    "Legal Consultation": {
        "power": 0.6,     # Legal authority
        "expertise": 0.9,  # Legal expertise
        "stakes": 0.9,     # Legal implications
        "emotional": 0.7,  # Legal concerns
        "formality": 0.9,  # Very formal
        "duration": 0.5    # Case duration
    },
    "Insurance Agent Explaining Coverage": {
        "power": 0.3,     # Professional guidance
        "expertise": 0.8,  # Insurance knowledge
        "stakes": 0.7,     # Coverage importance
        "emotional": 0.4,  # Technical discussion
        "formality": 0.7,  # Professional service
        "duration": 0.4    # Policy setup
    },
    "Career Counselor Session": {
        "power": 0.3,     # Guidance role
        "expertise": 0.7,  # Career knowledge
        "stakes": 0.8,     # Career impact
        "emotional": 0.6,  # Personal future
        "formality": 0.6,  # Professional but supportive
        "duration": 0.5    # Career planning
    },
    "Landlord-Tenant Discussion": {
        "power": 0.5,     # Property authority
        "expertise": 0.4,  # Property matters
        "stakes": 0.7,     # Living conditions
        "emotional": 0.6,  # Home issues
        "formality": 0.6,  # Professional but personal
        "duration": 0.7    # Lease duration
    },
    "College Admissions Interview": {
        "power": 0.8,     # Clear authority
        "expertise": 0.7,  # Admissions knowledge
        "stakes": 0.9,     # Educational future
        "emotional": 0.8,  # High stress
        "formality": 0.9,  # Very formal
        "duration": 0.1    # One-time event
    },
    "Tech Workshop": {
        "power": 0.5,     # Teaching role
        "expertise": 0.8,  # Technical knowledge
        "stakes": 0.5,     # Skill development
        "emotional": 0.2,  # Technical focus
        "formality": 0.6,  # Professional training
        "duration": 0.3    # Workshop duration
    },
    "Nutritionist Consultation": {
        "power": 0.4,     # Health guidance
        "expertise": 0.8,  # Nutrition expertise
        "stakes": 0.7,     # Health impact
        "emotional": 0.5,  # Health goals
        "formality": 0.6,  # Professional but personal
        "duration": 0.6    # Dietary planning
    },
    "Home Renovation Consultation": {
        "power": -0.2,    # Client-driven
        "expertise": 0.8,  # Design expertise
        "stakes": 0.8,     # Home investment
        "emotional": 0.6,  # Personal space
        "formality": 0.6,  # Professional service
        "duration": 0.5    # Project duration
    },
    "Volunteer Orientation": {
        "power": 0.4,     # Guidance role
        "expertise": 0.7,  # Organization knowledge
        "stakes": 0.4,     # Volunteer work
        "emotional": 0.3,  # Helpful atmosphere
        "formality": 0.5,  # Semi-formal
        "duration": 0.4    # Orientation period
    },
    "Pet Adoption Counseling": {
        "power": 0.4,     # Adoption authority
        "expertise": 0.7,  # Animal care knowledge
        "stakes": 0.8,     # Pet welfare
        "emotional": 0.7,  # Animal connection
        "formality": 0.5,  # Professional but warm
        "duration": 0.3    # Adoption process
    },
    "Online Dating Conversation": {
        "power": 0.0,     # Equal peers
        "expertise": 0.1,  # Personal interaction
        "stakes": 0.4,     # Potential relationship
        "emotional": 0.6,  # Personal connection
        "formality": 0.2,  # Very informal
        "duration": 0.2    # Initial contact
    },
    "Language Exchange": {
        "power": 0.3,     # Language expertise
        "expertise": 0.7,  # Native speaker knowledge
        "stakes": 0.4,     # Learning goals
        "emotional": 0.3,  # Supportive learning
        "formality": 0.4,  # Casual learning
        "duration": 0.5    # Learning period
    },
    "Coaching Session for Public Speaking": {
        "power": 0.5,     # Coach authority
        "expertise": 0.8,  # Speaking expertise
        "stakes": 0.7,     # Presentation success
        "emotional": 0.6,  # Performance anxiety
        "formality": 0.7,  # Professional training
        "duration": 0.4    # Training period
    },
    "Fitness Class Instruction": {
        "power": 0.6,     # Instructor authority
        "expertise": 0.8,  # Fitness expertise
        "stakes": 0.5,     # Exercise safety
        "emotional": 0.4,  # Physical activity
        "formality": 0.5,  # Professional but energetic
        "duration": 0.3    # Class duration
    },
    "Book Club Discussion": {
        "power": 0.2,     # Facilitator role
        "expertise": 0.3,  # Literary discussion
        "stakes": 0.2,     # Casual discussion
        "emotional": 0.5,  # Literary engagement
        "formality": 0.3,  # Casual gathering
        "duration": 0.4    # Club membership
    },
    "Tech Support for Smart Home Setup": {
        "power": 0.4,     # Technical expertise
        "expertise": 0.8,  # Technical knowledge
        "stakes": 0.5,     # Setup success
        "emotional": 0.5,  # Tech frustration
        "formality": 0.6,  # Professional support
        "duration": 0.2    # Setup period
    },
    "Online Gaming Teamwork": {
        "power": 0.3,     # Team leader role
        "expertise": 0.5,  # Game knowledge
        "stakes": 0.3,     # Game outcome
        "emotional": 0.6,  # Competitive spirit
        "formality": 0.2,  # Very informal
        "duration": 0.3    # Game duration
    },
    "Conflict Mediation": {
        "power": 0.5,     # Mediator authority
        "expertise": 0.7,  # Mediation skills
        "stakes": 0.8,     # Conflict resolution
        "emotional": 0.9,  # High tension
        "formality": 0.8,  # Professional process
        "duration": 0.4    # Resolution period
    },
    "Podcast Interview": {
        "power": 0.2,     # Host guidance
        "expertise": 0.5,  # Topic knowledge
        "stakes": 0.5,     # Public content
        "emotional": 0.4,  # Professional chat
        "formality": 0.5,  # Semi-formal
        "duration": 0.1    # One-time interview
    },
    "Environmental Activist Campaign": {
        "power": 0.3,     # Activist leadership
        "expertise": 0.7,  # Environmental knowledge
        "stakes": 0.8,     # Environmental impact
        "emotional": 0.7,  # Passionate cause
        "formality": 0.5,  # Professional but passionate
        "duration": 0.6    # Campaign period
    }
}

traits = ['E', 'A', 'C', 'N', 'O']

#output_folders = ["gpt4o/output_gpt4o_4"]
output_folders = ["gpt4o/output_gpt4o_4","llama70b/output_llama70b_1","llama8b/output_llama8b_3","phi/output_phi_1","qwen/output_qwen_1","tulu3/output_tulu_1"]
features = ["gpt4omini","llama70b","llama8b","phi4","qwen7b","tulu3","scenario_power", "scenario_expertise", "scenario_stakes", "scenario_emotional", "scenario_formality", "scenario_duration", "user_initial_E", "user_initial_A", "user_initial_C", "user_initial_N", "user_initial_O", "bot_initial_E", "bot_initial_A", "bot_initial_C", "bot_initial_N", "bot_initial_O", 
            "E_distance", "A_distance", "C_distance", "N_distance", "O_distance", "conversation_sentiment", "avg_user_reply", "avg_bot_reply"]
y_labels = ["bot_E_shift","bot_A_shift","bot_C_shift","bot_N_shift","bot_O_shift"]

openai.api_key = ''
def get_llm_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7,
        n=1
    )
    return response['choices'][0]['message']['content'].strip()

def get_sentiment_score(conversation):
    prompt = f'''Analyze the sentiment of the following conversation transcript. 
    Return a single numeric sentiment score with one decimal place between -1 and 1, where:
    -1 represents extremely negative sentiment
    - 0 represents neutral sentiment
    - 1 represents extremely positive sentiment

    Evaluate based on these criteria:
    1. Emotional tone of language
    2. Expressed satisfaction/dissatisfaction
    3. Presence of conflict or agreement
    4. Overall conversational dynamics

    Provide ONLY the numeric score. Do not include additional explanation.

    Conversation transcript:
    {conversation}'''

    try:
        llm_score = get_llm_response(prompt)
        print(llm_score)
        if len(llm_score) > 4:
            if llm_score[0] == '-':
                llm_score = llm_score[:4]
            else:
                llm_score = llm_score[:3]
        sentiment_score = float(llm_score)
    except Exception as e:
        print(f"Either failed to call OpenAI or convert number response at {file_path}")
        return None
    
    return sentiment_score

def extract_data(file_path, model_number):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    min_lines = 44 if model_number==0 else 48
    if len(lines) < min_lines:
        print(f"Minimum lines not met for {file_path}\n")
        return None
    
    row = []
    
    # LLM Model one hot encoding
    for i in range(6):
        row.append(1) if model_number==i else row.append(0)

    # Scenario variables
    scenario = lines[12][23:-1] if model_number!=0 else lines[10][23:-1]
    try:
        scenario_score = scenario_scores[scenario]
        row.append(scenario_score["power"])
        row.append(scenario_score["expertise"])
        row.append(scenario_score["stakes"])
        row.append(scenario_score["emotional"])
        row.append(scenario_score["formality"])
        row.append(scenario_score["duration"])
    except:
        print(f"Scenario matching failed for {file_path}")
        return None

    # Parse the initial scores
    for k in range(-2,-5,-2):
        try:
            # Use regex to find dictionary-like structure
            match = re.search(r'\{.*\}', lines[k])
            if match:
                # Evaluate the found dictionary string
                d = eval(match.group())
                # Check if dictionary has all required keys
                if all(key in d for key in ['E', 'A', 'C', 'N', 'O']):
                    #d['N'] = 76 - d['N'] # Correcting wrong N to ES = 40 - (score + 4 - 40)
                    row.append(d['E'])
                    row.append(d['A'])
                    row.append(d['C'])
                    row.append(d['N'])
                    row.append(d['O'])
                else:
                    return None
            else:
                print(f"Failed to parse initial scores for {file_path}")
                return None
        except:
            return None
    
    # Initial personality distance, calculated user - bot
    row.append(row[12]-row[17])
    row.append(row[13]-row[18])
    row.append(row[14]-row[19])
    row.append(row[15]-row[20])
    row.append(row[16]-row[21])

    start, end = 0, 0
    if model_number==0:
        start = 10
        end = -13
    else:
        start = 12
        end = -15
    conversation = '\n'.join(''.join(line) for line in lines[start:end])

    # Conversation sentiment
    sentiment_score = get_sentiment_score(conversation)
    if sentiment_score:
        row.append(sentiment_score)
    else:
        print(f"Sentiment not accepted for {model_number} at {file_path}\n")
        return None
    
    # Average user and bot reply lengths
    user_character_count, bot_character_count = 0, 0
    i = 1
    prevA = True
    for line in lines[start:end]:
        if len(line) < 15:
            continue
        if prevA: #looking for user's/B's turn
            desiredStart = f"LLM B (Turn {i})"
            if line.startswith(desiredStart):
                user_character_count += len(line) - 16
                prevA = False
            else:
                continue #Could be a fake response for other LLM
        else: #looking for chatbot's/A's turn
            desiredStart = f"LLM A (Turn {i})"
            if line.startswith(desiredStart):
                bot_character_count += len(line) - 16
                prevA = True
                i += 1
            else:
                continue #Could be a fake response for other LLM
    user_character_count /= 10
    bot_character_count /= 10
    row.append(user_character_count)
    row.append(bot_character_count)

    # Outputs
    try:
        match = re.search(r'\{.*\}', lines[-3])
        if match:
            d = eval(match.group())
            if all(key in d for key in ['E', 'A', 'C', 'N', 'O']):
                row.append(d['E'])
                row.append(d['A'])
                row.append(d['C'])
                row.append(d['N'])
                row.append(d['O'])
            else:
                return None
        else:
            return None
    except:
        return None    

    return row if len(row) == len(features) + len(y_labels) else None

data = [features]
y = [y_labels]

for i in range(1,6):
    output_folder = output_folders[i]
    for j in range(1, 1001):
        file_path = f"{output_folder}/output_{j}.txt"
        if os.path.exists(file_path):
            result = extract_data(file_path, i)
            if result:
                data.append(result[:30])
                y.append(result[30:])
            else:
                print(f"No row appended for {file_path}")


with open("features.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

with open("outputs.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(y)

print("Extraction completed")