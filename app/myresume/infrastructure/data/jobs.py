"""The job postings searched by the lab (a fixed, hand-written corpus so evaluation results are reproducible)."""

from __future__ import annotations

from app.myresume.domain.models import Job
from app.myresume.infrastructure.data.job_details import DETAILS


def _job(id: str, title: str, company: str, location: str, mode: str, level: str, summary: str, skills: str) -> Job:
    d = DETAILS[id]
    return Job(
        id=id, title=title, company=company, location=location, work_mode=mode, level=level, summary=summary,
        skills=[s.strip() for s in skills.split(",")], experience=d.experience, responsibilities=d.responsibilities,
        qualifications=d.qualifications, nice_to_have=d.nice_to_have,
    )


JOBS: list[Job] = [
    # ---- software engineering
    _job("backend-python", "Senior Backend Engineer (Python)", "Northwind Payments", "Berlin", "Hybrid", "Senior",
         "Design and build the REST APIs and services behind our payments platform. You will own services end to end, "
         "from schema design to deployment, and improve reliability and performance of production systems.",
         "Python, FastAPI, Django, PostgreSQL, REST APIs, Docker, AWS, unit testing"),
    _job("backend-java", "Java Backend Developer", "Helvetia Bank", "Zurich", "On-site", "Mid-level",
         "Develop microservices for core banking in a regulated environment. You will implement event-driven "
         "workflows and keep transaction processing correct, observable and secure.",
         "Java, Spring Boot, Kafka, microservices, SQL, JUnit, Maven, Kubernetes"),
    _job("backend-go", "Go Platform Engineer", "Cloudlane", "Amsterdam", "Remote", "Senior",
         "Build the internal platform that other teams deploy onto. You will write high-throughput services and "
         "tooling in Go and operate them on container orchestration.",
         "Go, gRPC, Kubernetes, distributed systems, Redis, Prometheus, Linux"),
    _job("frontend-react", "Frontend Engineer (React)", "Pixelworks", "London", "Hybrid", "Mid-level",
         "Create fast, accessible user interfaces for our analytics dashboard. You will turn designs into reusable "
         "component libraries and care about performance and a11y.",
         "React, TypeScript, Redux, CSS, HTML, Jest, Storybook, accessibility"),
    _job("frontend-angular", "Angular Developer", "Brightside Insurance", "Madrid", "Hybrid", "Mid-level",
         "Maintain and extend the customer portal built with Angular. You will work with designers and back-end "
         "engineers to ship new self-service features.",
         "Angular, TypeScript, RxJS, SCSS, REST integration, Cypress"),
    _job("fullstack-node", "Full Stack Developer (Node.js + React)", "Launchpad Studio", "Lisbon", "Remote", "Mid-level",
         "Ship product features across the stack for a fast-moving startup. You will build React screens and the "
         "Node.js APIs that feed them, and deploy to the cloud.",
         "Node.js, Express, React, TypeScript, MongoDB, REST APIs, Git, AWS"),
    _job("ios-engineer", "iOS Engineer", "Tempo Fitness", "Munich", "Hybrid", "Mid-level",
         "Develop and maintain our iPhone app used by two million members. You will build native screens, integrate "
         "HealthKit and publish releases to the App Store.",
         "Swift, SwiftUI, UIKit, Xcode, Core Data, HealthKit, App Store releases"),
    _job("android-engineer", "Android Engineer", "Wayfinder Maps", "Warsaw", "Remote", "Mid-level",
         "Build the Android version of our navigation app. You will write modern Kotlin code, optimise battery use "
         "and keep the app stable across many devices.",
         "Kotlin, Jetpack Compose, Android SDK, Coroutines, Room, Gradle, Play Store"),
    _job("flutter-developer", "Cross-platform Mobile Developer (Flutter)", "Marketly", "Remote", "Remote", "Mid-level",
         "Deliver one codebase for iOS and Android shopping apps. You will build UI, state management and native "
         "integrations with Flutter.",
         "Flutter, Dart, REST APIs, state management, Firebase, iOS, Android"),
    _job("devops-engineer", "DevOps Engineer", "Stackforge", "Dublin", "Hybrid", "Mid-level",
         "Automate how software is built, tested and released. You will manage cloud infrastructure as code, "
         "maintain delivery pipelines and run our Kubernetes clusters.",
         "Terraform, Kubernetes, Docker, CI/CD, Jenkins, GitHub Actions, AWS, Linux, monitoring"),
    _job("sre", "Site Reliability Engineer", "Streamly", "Remote", "Remote", "Senior",
         "Keep a streaming service available for millions of viewers. You will define service level objectives, "
         "build observability, respond to incidents and lead blameless postmortems.",
         "SLOs, incident response, on-call, observability, Prometheus, Grafana, Linux, Python, automation"),
    _job("cloud-architect", "Cloud Solutions Architect", "Altus Consulting", "Paris", "Hybrid", "Senior",
         "Advise enterprise customers on moving workloads to the cloud. You will design secure, cost-efficient "
         "architectures and guide migration projects.",
         "AWS, Azure, solution architecture, networking, cost optimisation, security, migration, stakeholder communication"),
    _job("security-engineer", "Application Security Engineer", "Fortis Systems", "Remote", "Remote", "Senior",
         "Make our products secure by design. You will review code and architecture, run penetration tests and "
         "build security checks into the delivery pipeline.",
         "application security, threat modelling, OWASP, penetration testing, SAST, DevSecOps, Python"),
    _job("qa-automation", "QA Automation Engineer", "Testwise", "Prague", "Hybrid", "Mid-level",
         "Build the automated test suites that guard every release. You will write end-to-end and API tests and "
         "improve the test strategy with developers.",
         "Selenium, Cypress, Python, API testing, test automation, CI, Postman, Agile"),
    _job("embedded-engineer", "Embedded Firmware Engineer", "Voltia Devices", "Stuttgart", "On-site", "Mid-level",
         "Write firmware for battery-powered sensors. You will work close to the hardware, debug with oscilloscopes "
         "and optimise memory and power usage.",
         "C, C++, RTOS, microcontrollers, I2C, SPI, debugging, low power design"),
    _job("game-developer", "Unity Game Developer", "Brightpixel Games", "Helsinki", "Hybrid", "Mid-level",
         "Build gameplay features for a mobile puzzle game. You will script mechanics, profile performance and "
         "collaborate with artists and designers.",
         "Unity, C#, gameplay programming, mobile games, profiling, 3D maths"),
    _job("database-admin", "Database Administrator", "Datacore Solutions", "Vienna", "On-site", "Senior",
         "Run our production PostgreSQL and Oracle databases. You will tune queries, manage backups and replication "
         "and plan capacity.",
         "PostgreSQL, Oracle, SQL tuning, backup and recovery, replication, high availability, Linux"),
    # ---- data and AI
    _job("data-scientist", "Data Scientist", "Insightly", "Remote", "Remote", "Mid-level",
         "Use statistics and machine learning to answer business questions. You will build forecasting and "
         "classification models, design experiments and present findings to stakeholders.",
         "Python, pandas, scikit-learn, statistics, A/B testing, SQL, forecasting, data visualisation"),
    _job("ml-engineer", "Machine Learning Engineer", "Neuralis", "Zurich", "Hybrid", "Senior",
         "Take models from notebook to production. You will train deep learning models, build training pipelines "
         "and serve models at scale with monitoring.",
         "PyTorch, Python, MLOps, model deployment, Docker, feature engineering, Kubernetes, monitoring"),
    _job("llm-engineer", "LLM / NLP Engineer", "Lexa AI", "Remote", "Remote", "Senior",
         "Build assistants powered by large language models. You will design retrieval-augmented generation "
         "pipelines, evaluate answer quality and fine-tune language models.",
         "LLMs, RAG, LangChain, embeddings, vector databases, prompt engineering, Python, NLP, evaluation"),
    _job("cv-engineer", "Computer Vision Engineer", "Sightline Robotics", "Gothenburg", "On-site", "Senior",
         "Teach robots to see. You will develop object detection and segmentation models and deploy them on edge "
         "devices.",
         "computer vision, PyTorch, OpenCV, object detection, CUDA, C++, edge deployment"),
    _job("data-engineer", "Data Engineer", "Flowdata", "Berlin", "Hybrid", "Mid-level",
         "Build the pipelines that move and transform data for analytics. You will orchestrate batch and streaming "
         "jobs and keep the warehouse reliable.",
         "Spark, Airflow, SQL, Python, ETL, data warehouse, dbt, Kafka, cloud storage"),
    _job("data-analyst", "Data Analyst", "Retailo", "Barcelona", "Hybrid", "Junior",
         "Turn sales and customer data into insight. You will write SQL, build dashboards and report trends to "
         "business teams.",
         "SQL, Excel, Tableau, data visualisation, reporting, statistics, Python"),
    _job("bi-developer", "Business Intelligence Developer", "Ledgerline", "Milan", "On-site", "Mid-level",
         "Design the reports and data models executives rely on. You will build semantic models and dashboards in "
         "Power BI on top of the data warehouse.",
         "Power BI, DAX, SQL, data modelling, ETL, reporting, stakeholder requirements"),
    _job("analytics-engineer", "Analytics Engineer", "Metricly", "Remote", "Remote", "Mid-level",
         "Own the clean, tested datasets analysts build on. You will model data with SQL and dbt and document "
         "metrics so everyone uses the same definitions.",
         "dbt, SQL, data modelling, Git, Looker, data quality testing, Snowflake"),
    # ---- product and design
    _job("product-manager", "Product Manager", "Orbit Software", "Remote", "Remote", "Senior",
         "Own the roadmap of a B2B product. You will talk to customers, prioritise the backlog, define success "
         "metrics and work with engineering and design to deliver.",
         "product roadmap, user research, prioritisation, OKRs, agile, stakeholder management, analytics"),
    _job("ux-designer", "UX / Product Designer", "Craftsman Labs", "Copenhagen", "Hybrid", "Mid-level",
         "Design intuitive experiences for web and mobile apps. You will run user research, build prototypes and "
         "hand off polished designs to developers.",
         "Figma, user research, wireframing, prototyping, usability testing, design systems, interaction design"),
    _job("project-manager", "Technical Project Manager", "Delivera", "Dublin", "Hybrid", "Senior",
         "Keep software projects on time and on budget. You will plan sprints, remove blockers, manage risks and "
         "report progress to stakeholders.",
         "project planning, Scrum, Jira, risk management, budgeting, stakeholder communication, agile coaching"),
    _job("business-analyst", "Business Analyst", "Clearpath Finance", "London", "Hybrid", "Mid-level",
         "Bridge the business and the IT teams. You will gather requirements, document processes and write "
         "user stories that engineers can build.",
         "requirements gathering, process modelling, user stories, SQL, Jira, stakeholder workshops"),
    # ---- business functions
    _job("digital-marketing", "Digital Marketing Manager", "Greenleaf Commerce", "Remote", "Remote", "Mid-level",
         "Grow traffic and conversions for an online store. You will plan paid and organic campaigns, manage SEO "
         "and email automation and analyse channel performance.",
         "SEO, Google Analytics, Google Ads, email marketing, content strategy, campaign management, social media"),
    _job("content-writer", "Technical Content Writer", "Devpress", "Remote", "Remote", "Mid-level",
         "Write clear articles and documentation for developers. You will explain complex topics, edit contributions "
         "and keep our docs accurate.",
         "technical writing, editing, documentation, SEO writing, Markdown, API documentation, research"),
    _job("sales-executive", "Account Executive (SaaS)", "Salesloop", "New York", "Hybrid", "Mid-level",
         "Close new business for a software product. You will run demos, negotiate contracts and hit quarterly "
         "revenue targets.",
         "B2B sales, CRM, negotiation, pipeline management, demos, prospecting, Salesforce"),
    _job("customer-success", "Customer Success Manager", "Supportly", "Remote", "Remote", "Mid-level",
         "Help customers get value from our platform. You will onboard accounts, run business reviews and reduce "
         "churn.",
         "customer onboarding, account management, churn reduction, communication, CRM, renewals"),
    _job("technical-recruiter", "Technical Recruiter", "Talentbridge", "Remote", "Remote", "Mid-level",
         "Hire engineers for fast-growing companies. You will source candidates, run screening calls and guide "
         "hiring managers.",
         "sourcing, interviewing, applicant tracking systems, employer branding, stakeholder management"),
    _job("financial-analyst", "Financial Analyst", "Capitalis", "Frankfurt", "On-site", "Mid-level",
         "Support planning with financial models and reports. You will analyse budgets, forecast revenue and "
         "explain variances.",
         "financial modelling, Excel, forecasting, budgeting, variance analysis, SQL, reporting"),
    _job("accountant", "Staff Accountant", "Baker & Reed", "Chicago", "On-site", "Junior",
         "Keep the books accurate. You will process invoices, reconcile accounts and prepare monthly closings.",
         "bookkeeping, accounts payable, reconciliation, month-end close, QuickBooks, GAAP"),
    # ---- outside tech
    _job("registered-nurse", "Registered Nurse", "St. Anne Hospital", "Leeds", "On-site", "Mid-level",
         "Provide bedside care on a medical ward. You will assess patients, administer medication and coordinate "
         "with doctors and families.",
         "patient care, medication administration, clinical assessment, teamwork, electronic health records"),
    _job("mechanical-engineer", "Mechanical Design Engineer", "Aerowerk", "Toulouse", "On-site", "Mid-level",
         "Design and test mechanical components for aircraft systems. You will model parts, run simulations and "
         "support manufacturing.",
         "CAD, SolidWorks, finite element analysis, GD&T, materials, prototyping"),
    _job("supply-chain", "Supply Chain Analyst", "Logistix", "Rotterdam", "Hybrid", "Mid-level",
         "Optimise how goods flow from supplier to customer. You will forecast demand, track inventory and "
         "improve delivery times.",
         "demand forecasting, inventory management, Excel, SQL, ERP, logistics, analytics"),
    _job("graphic-designer", "Graphic Designer", "Hue & Co", "Remote", "Remote", "Mid-level",
         "Create brand visuals for campaigns and social media. You will design layouts, illustrations and "
         "presentations.",
         "Adobe Illustrator, Photoshop, typography, branding, layout, social media graphics"),
    _job("hr-generalist", "HR Generalist", "Peoplefirst", "Madrid", "On-site", "Mid-level",
         "Support employees through the whole lifecycle. You will handle onboarding, payroll questions and "
         "employee relations.",
         "onboarding, employee relations, HR policies, payroll basics, communication, labour law"),
]

JOBS_BY_ID: dict[str, Job] = {j.id: j for j in JOBS}
assert len(JOBS_BY_ID) == len(JOBS), "duplicate job ids"
assert set(DETAILS) == set(JOBS_BY_ID), "every job needs details and every detail a job"
