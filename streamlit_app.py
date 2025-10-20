"""
IPE Job Description Generator & Evaluator - REVISED VERSION
============================================================

This tool generates and evaluates job descriptions using Mercer IPE framework.
Major improvements:
- Uses official Mercer IPE definitions (not simplified versions)
- Proper guardrails for all four dimensions (Impact, Contribution, Innovation, Knowledge)
- Few-shot learning with calibration examples
- Fractional team responsibility support (1, 1.5, 2, 2.5, 3)
- Detailed audit trails showing why each rating was assigned
- Better distinction between functional vs. strategic scope
"""

import streamlit as st
import pandas as pd
import requests
import os
import json
import re
from typing import Dict, Tuple, Optional, List

###############################
# Configuration              #
###############################
st.set_page_config(
    page_title="IPE Job Description Tool (Revised)",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLAUDE_MODEL = "claude-sonnet-4-20250514"  # Cost-effective and highly capable
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

###############################
# IPE Framework Data         #
###############################

# Official Mercer IPE Definitions with organizational context

IMPACT_DEFINITIONS = """
IMPACT DIMENSION (Rows in scoring table):

1. DELIVERY - Very low positions in professional career stream
   Area of Impact: Job Area | Type of Impact: Within specific standards and guidelines
   Definition: Deliver own output by following defined procedures/processes under close supervision and guidance.
   Contribution context: Limited impact, hard to discern contribution to achievement of results.

2. OPERATION - Supervisor in mgmt, professional career stream
   Area of Impact: Job Area(s) | Type of Impact: Within operational targets or service standards
   Definition: Work to achieve objectives and deliver results with a short-term, operational focus and limited impact on others.
   Contribution context: Work achieves operational targets; some impact on others within job area.

3. TACTICAL - Most positions in mgmt expert/specialist positions in professional career stream
   Area of Impact: Business Function | Type of Impact: New products/processes/standards based on organizational strategy
   Definition: Develop new products, processes, standards or operational plans in support of organization's business strategies.
   Key distinction: Impact at FUNCTION level (16-20%), not across multiple functions.
   Contribution context: Marked contribution to defining direction for new products/processes/standards.

4. STRATEGIC - Division Level - most executive levels
   Area of Impact: Organization level | Type of Impact: Longer-term plans based on organization vision
   Definition: Directly influences development of corporate business unit or organization's business strategies.
   Key distinction: Impact at DIVISION/BU level (15-20%), involvement in enterprise-level strategy.
   Contribution context: Marked contribution to defining business strategies of corporate business unit.

5. VISIONARY - Corporate level - Group Level - Most executive levels
   Area of Impact: Corporate level | Type of Impact: Vision, mission & value
   Definition: Lead an organization within a corporation or multiple organizations/BUs; freedom to define vision and direction.
   Key distinction: Impact at CORPORATE/GROUP level (30%+), defines organizational mission and values.
   Contribution context: Predominant authority for defining business strategies; major influence on overall results.
"""

CONTRIBUTION_DEFINITIONS = """
CONTRIBUTION DIMENSION (Columns in Impact/Contribution table):

1. LIMITED - Hard to identify/discern contribution to achievement of results
   Definition: Deliver own output by following defined procedures/processes under close supervision and guidance.

2. SOME - Easily discernible or measurable contribution that usually leads indirectly to achievement of results
   Definition: Deliver own output following broad framework or standards with some impact on job area.

3. DIRECT - Directly and clearly influences the course of action that determines the achievement of results
   Definition: Deliver own output to meet specific operational targets; direct, visible contribution within job area.

4. SIGNIFICANT - Quite marked contribution with authority of a frontline or primary nature
   Definition: Deliver own output within broad operational targets; significant impact within job area context.

5. MAJOR - Predominant authority in determining the achievement of key results
   Definition: Deliver own output with major impact on broader operational targets; predominant influence on key results.

IMPORTANT: Contribution must be assessed in context of Impact level.
Example: At Tactical (Level 3), a "Significant" contribution means 20-30% impact on function results.
At Strategic (Level 4), "Significant" means 20-30% impact on BU/organization results.
"""

COMMUNICATION_DEFINITIONS = """
COMMUNICATION DIMENSION (Rows in Communication/Frame table):

1. CONVEY - Communicate information by statement, suggestion, gesture, or appearance
   Nature: Statements, suggestions | Desired Outcome: Understanding information | Frequency: 1, Frequent & Continuous
   Definition: Obtain and provide information to others within the organization (or external parties depending on Frame).

2. ADAPT AND EXCHANGE - Reach agreement through flexibility and compromise
   Nature: Reach agreement through flexibility and compromise | Desired Outcome: Comprehension of facts/policies
   Frequency: 1.5 Occasional, 2 Frequent, 2.5 Continuous
   Definition: Explain facts, practices, policies, etc. to others through flexibility and compromise.

3. INFLUENCE - Effect change without direct exercise of command where persuasion is required
   Nature: Effect change without commands | Desired Outcome: Acceptance of new concepts, practices, approach
   Frequency: 2.5 Occasional, 3 Frequent, 3.5 Continuous
   Definition: Convince others where strong interest exists (or skepticism/resistance exists depending on Frame).

4. NEGOTIATE - Come to agreement by managing communications through discussions and compromise
   Nature: Strategic short-term communication | Desired Outcome: Acceptance through discussions and compromise
   Frequency: 3.5 Occasional, 4 Frequent, 4.5 Continuous
   Definition: Convince others to accept complete proposals and programs where interest in cooperation varies.

5. NEGOTIATE LONG-TERM - Manage communications of great importance having long-term, strategic implications
   Nature: Strategic long-term communication | Desired Outcome: Acceptance of strategic agreement
   Frequency: 4.5 Occasional, 5 Frequent
   Definition: Reach agreement of strategic importance with others who have differing points of view.
"""

FRAME_DEFINITIONS = """
FRAME DIMENSION (Columns in Communication/Frame table):

1. INTERNAL SHARED INTERESTS (Frame 1.0 point)
   Definition: Common desire to reach solution within a corporation.
   Applies to: Finance, Engineering, Logistics functions (Frame 1).
   Example: Communication within accounting team on new procedures.

2. EXTERNAL SHARED INTERESTS (Frame 2.0 points)
   Definition: Common desire to reach solution outside a corporation.
   Applies to: Purchasing, PR, Marketing positions (Frame 2).
   Example: Vendor negotiations with mutual interest in agreement.

3. INTERNAL DIVERGENT INTERESTS (Frame 3.0 points)
   Definition: Conflicting objectives that inhibit reaching a solution within a corporation.
   Applies to: HR, Labor Relations (Frame 3).
   Example: Internal negotiation between departments with competing priorities.

4. EXTERNAL DIVERGENT INTERESTS (Frame 4.0 points)
   Definition: Conflicting objectives that inhibit reaching a solution outside a corporation.
   Applies to: Sales, Legal (Frame 4).
   Example: Contract negotiation with external party with different interests.

KEY: Frame level determines communication baseline difficulty.
Internal/Shared = easier communication. External/Divergent = harder communication.
"""

INNOVATION_DEFINITIONS = """
INNOVATION DIMENSION (Rows in Innovation/Complexity table):

1. FOLLOW - Compare with source or authority; no changes expected
   Definition: Follow a set procedure in performance of repeated tasks or job activities.
   Organizational context: 1.5-1.6 Experienced/Entry, Specialist, Individual contributors

2. CHECK - Make minor changes
   Definition: Check and correct problems that are not immediately evident in existing systems or process.
   Organizational context: 2-2.5 Single function mgr, Expert, Supervisors

3. MODIFY - Adapt or enhance quality or value in existing methods
   Definition: Identify problems and update or modify working methods in own role without the benefit of defined procedures.
   Organizational context: 2.5-3-3.5 Head of org, Function head, Senior & experienced level positions

4. IMPROVE - Change significantly by enhancing entire existing processes, systems or products
   Definition: Analyze complex issues and significantly improve, change or adapt existing methods and techniques.
   Organizational context: 3.5-4 CEO, COO, Head of Division, Head of Region, Head of BU (Multi-dimensional)

5. CREATE/CONCEPTUALIZE - Develop truly new concepts or methods that break new ground
   Definition: Analyze complex issues before creating/conceptualizing truly new methods across job areas or functions.
   Organizational context: 4+ CEO, COO, Head of Division, Head of Region, Head of BU

6. SCIENTIFIC/TECHNICAL BREAKTHROUGH - Form and bring into existence major new advances
   Definition: Bring together multiple concepts across functions to define new direction or significant advance.
   Organizational context: 4+ CEO, COO, Head of Division, Head of Region, Head of BU
"""

COMPLEXITY_DEFINITIONS = """
COMPLEXITY DIMENSION (Columns in Innovation/Complexity table):

1. DEFINED - Single job area or discipline; scope of problem is well-defined
   Definition: Problems and issues generally fall within a single job area or discipline with well-defined scope.
   Example: Accounting analyst working within accounting standards.

2. DIFFICULT - Vaguely defined; requires understanding other disciplines
   Definition: Problems and issues may be only vaguely defined and require understanding of other disciplines/job areas.
   Example: Controller coordinating between accounting and operations.

3. COMPLEX - Broad-based solutions requiring two of three dimensions
   Definition: Problems require broad-based solutions requiring consideration of TWO of three dimensions: Operational, Financial, Human.
   Example: Sr Controller designing processes affecting operations AND financials.

4. MULTI-DIMENSIONAL - End-to-end solutions with all three dimensions
   Definition: Problems are truly multi-dimensional, requiring end-to-end solutions with direct impact on all three dimensions.
   Example: Division head addressing operational efficiency, financial optimization, AND organizational structure.

KEY BOUNDARY: Most within-function roles operate at Complexity 1-2. Cross-functional work = Complexity 2-3.
End-to-end organizational impact = Complexity 4.
"""

KNOWLEDGE_DEFINITIONS = """
KNOWLEDGE DIMENSION (Rows in Knowledge/Teams table):

1. LIMITED JOB KNOWLEDGE - Primary school
   Experience: No previous experience | Application: Basic work routines within narrow boundaries.

2. BASIC JOB KNOWLEDGE - Specialized School
   Experience: 0-6 months relevant | Application: Specialized knowledge of specific technical or office operations.

3. BROAD JOB KNOWLEDGE - Specialized degree + 6 months to 2 years
   Experience: 6 months to 2 years | Application: Broader knowledge of theory and principles within professional discipline.

4. EXPERTISE - University Degree + 2-5 years
   Experience: 2-5 years progressively responsible | Application: Deep knowledge of one job area or broad knowledge of several.

5. PROFESSIONAL STANDARD - University Degree + 5-8 years
   Experience: 5-8 years deep or cross-disciplinary | Application: Mastery of specific discipline + deep organizational knowledge.

6. ORG. GENERALIST / FUNCTIONAL SPECIALIST - University Degree + 8-12 years
   Experience: 8-12 years significant impact roles | Application: Concentrated knowledge in particular discipline OR broad knowledge across several areas.

7. BROAD PRACTICAL EXPERIENCE / FUNCTIONAL PREEMINENCE - University Degree + 12-16 years
   Experience: 12-16 years enterprise-level influence | Application: Preeminent expertise across functions OR broad experience in major functions.

8. BROAD AND DEEP PRACTICAL EXPERIENCE - University Degree + 16+ years
   Experience: 16+ years enterprise leadership | Application: Very significant experience across multiple businesses and functions.

KEY BOUNDARY: Entry-level white-collar (bachelor's degree, 0 years experience) = Knowledge 3-4, NOT Knowledge 1-2.
Knowledge 1-2 typically for blue-collar or vocational roles.
"""

TEAMS_DEFINITIONS = """
TEAMS DIMENSION (Columns in Knowledge/Teams table):

1. TEAM MEMBER - Individual contributor, no direct responsibility for leading others
   Definition: Individual contributor with no formal management responsibilities.
   Example: Staff accountant, software engineer, analyst.

1.5. HYBRID - Limited leadership or project leadership without formal reports
   Definition: Leads projects or coordinates activities but no formal direct reports.
   Example: Project lead, team coordinator, senior individual contributor.

2. TEAM LEADER - Coaches team members (at least three) in skills; leads, schedules, allocates and monitors work
   Definition: Formally responsible for leading a small team, typically 3-8 people.
   Example: Supervisor, team manager.

2.5. HYBRID MANAGER - Partial people management with significant IC responsibilities
   Definition: Manages smaller team while maintaining significant individual contributor role.

3. TEAMS MANAGER - Directs more than one team; determines team structure and roles of members
   Definition: Manages multiple teams or larger organization; sets structure and strategy.
   Example: Department manager, function head.
"""

BREADTH_DEFINITIONS = """
BREADTH DIMENSION (Separate scoring):

1. DOMESTIC - Domestic role
   Definition: Applies knowledge within a country or neighbouring countries with similar culture.
   Points: 0

1.5. SUB-REGION - Sub-region role
   Definition: Applies knowledge in a continental region (e.g. Europe, Asia, North America, Latin America, Middle East).
   Points: 5

2. REGIONAL - Regional role
   Definition: Applies knowledge across a continental region with diverse cultures.
   Points: 10

2.5. MULTIPLE REGIONS - Multiple regions role
   Definition: Minimum two continental regions represented.
   Points: 15

3. GLOBAL - Global role
   Definition: Applies knowledge across all regions of the world.
   Points: 20
"""

###############################
# Lookup Tables              #
###############################

# Impact/Contribution points table (intermediate lookup)
IMPACT_CONTRIBUTION_TABLE = {
    1: {1: 1, 1.5: 1.5, 2: 2, 2.5: 2.5, 3: 3, 3.5: 3.5, 4: 4, 4.5: 4.5, 5: 5},
    1.5: {1: 2.5, 2: 3, 2.5: 3.5, 3: 4, 3.5: 4.5, 4: 5, 4.5: 5.5, 5: 6.5},
    2: {1: 4, 1.5: 4.5, 2: 5, 2.5: 5.5, 3: 6, 3.5: 6.5, 4: 7, 4.5: 7.5, 5: 8},
    2.5: {1: 5.5, 1.5: 6, 2: 6.5, 2.5: 7, 3: 7.5, 3.5: 8, 4: 8.5, 4.5: 9, 5: 9.5},
    3: {1: 7, 1.5: 7.5, 2: 8, 2.5: 8.5, 3: 9, 3.5: 9.5, 4: 10, 4.5: 10.5, 5: 11},
    3.5: {1: 8.5, 1.5: 9, 2: 9.5, 2.5: 10, 3: 10.5, 3.5: 11, 4: 11.5, 4.5: 12, 5: 12.5},
    4: {1: 10, 1.5: 10.5, 2: 11, 2.5: 11.5, 3: 12, 3.5: 12.5, 4: 13, 4.5: 13.5, 5: 14},
    4.5: {1: 11.5, 1.5: 12, 2: 12.5, 2.5: 13, 3: 13.5, 3.5: 14, 4: 14.5, 4.5: 15, 5: 15.5},
    5: {1: 13, 1.5: 13.5, 2: 14, 2.5: 14.5, 3: 15, 3.5: 15.5, 4: 16, 4.5: 16.5, 5: 17},
}

# Impact/Size points table (uses intermediate value from Impact/Contribution)
IMPACT_SIZE_TABLE = {
    # Row = intermediate value from Impact/Contribution, Column = Size
    1: {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5},
    1.5: {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10, 11: 10, 12: 10, 13: 10},
    2: {1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 15, 9: 15, 10: 15, 11: 15, 12: 15, 13: 15},
    2.5: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 20, 9: 20, 10: 20, 11: 20, 12: 20, 13: 20},
    3: {1: 25, 2: 25, 3: 25, 4: 25, 5: 25, 6: 25, 7: 25, 8: 25, 9: 25, 10: 25, 11: 25, 12: 25, 13: 25},
    3.5: {1: 31, 2: 32, 3: 32, 4: 33, 5: 33, 6: 34, 7: 34, 8: 35, 9: 35, 10: 36, 11: 36, 12: 37, 13: 37},
    4: {1: 37, 2: 38, 3: 39, 4: 40, 5: 41, 6: 42, 7: 43, 8: 44, 9: 45, 10: 46, 11: 47, 12: 48, 13: 49},
    4.5: {1: 41, 2: 43, 3: 44, 4: 46, 5: 47, 6: 49, 7: 50, 8: 52, 9: 53, 10: 55, 11: 56, 12: 58, 13: 59},
    5: {1: 44, 2: 46, 3: 48, 4: 50, 5: 52, 6: 54, 7: 56, 8: 58, 9: 60, 10: 62, 11: 64, 12: 66, 13: 68},
    5.5: {1: 50, 2: 53, 3: 55, 4: 58, 5: 60, 6: 63, 7: 65, 8: 68, 9: 70, 10: 73, 11: 75, 12: 78, 13: 80},
    6: {1: 56, 2: 59, 3: 62, 4: 65, 5: 68, 6: 71, 7: 74, 8: 77, 9: 80, 10: 83, 11: 86, 12: 89, 13: 92},
    6.5: {1: 60, 2: 64, 3: 67, 4: 71, 5: 74, 6: 78, 7: 81, 8: 85, 9: 88, 10: 92, 11: 95, 12: 99, 13: 102},
    7: {1: 63, 2: 67, 3: 71, 4: 75, 5: 79, 6: 83, 7: 87, 8: 91, 9: 95, 10: 99, 11: 103, 12: 107, 13: 111},
    7.5: {1: 72, 2: 76, 3: 80, 4: 85, 5: 89, 6: 93, 7: 97, 8: 102, 9: 106, 10: 110, 11: 114, 12: 119, 13: 123},
    8: {1: 80, 2: 85, 3: 89, 4: 94, 5: 98, 6: 103, 7: 107, 8: 112, 9: 116, 10: 121, 11: 125, 12: 130, 13: 134},
    8.5: {1: 84, 2: 89, 3: 94, 4: 99, 5: 104, 6: 109, 7: 114, 8: 119, 9: 124, 10: 129, 11: 134, 12: 139, 13: 144},
    9: {1: 87, 2: 93, 3: 98, 4: 104, 5: 109, 6: 115, 7: 120, 8: 126, 9: 131, 10: 137, 11: 142, 12: 148, 13: 153},
    9.5: {1: 96, 2: 102, 3: 107, 4: 113, 5: 119, 6: 125, 7: 130, 8: 136, 9: 142, 10: 148, 11: 153, 12: 159, 13: 165},
    10: {1: 104, 2: 110, 3: 116, 4: 122, 5: 128, 6: 134, 7: 140, 8: 146, 9: 152, 10: 158, 11: 164, 12: 170, 13: 176},
    10.5: {1: 108, 2: 115, 3: 121, 4: 128, 5: 134, 6: 141, 7: 147, 8: 154, 9: 160, 10: 167, 11: 173, 12: 180, 13: 186},
    11: {1: 111, 2: 118, 3: 125, 4: 132, 5: 139, 6: 146, 7: 153, 8: 160, 9: 167, 10: 174, 11: 181, 12: 188, 13: 195},
    11.5: {1: 120, 2: 128, 3: 135, 4: 143, 5: 150, 6: 158, 7: 165, 8: 173, 9: 180, 10: 188, 11: 195, 12: 203, 13: 210},
    12: {1: 128, 2: 136, 3: 144, 4: 152, 5: 160, 6: 168, 7: 176, 8: 184, 9: 192, 10: 200, 11: 208, 12: 216, 13: 224},
    12.5: {1: 132, 2: 141, 3: 149, 4: 158, 5: 166, 6: 175, 7: 183, 8: 192, 9: 200, 10: 209, 11: 217, 12: 226, 13: 234},
    13: {1: 135, 2: 144, 3: 153, 4: 162, 5: 171, 6: 180, 7: 189, 8: 198, 9: 207, 10: 216, 11: 225, 12: 234, 13: 243},
    13.5: {1: 141, 2: 151, 3: 160, 4: 170, 5: 179, 6: 189, 7: 198, 8: 208, 9: 217, 10: 227, 11: 236, 12: 246, 13: 255},
    14: {1: 147, 2: 157, 3: 167, 4: 177, 5: 187, 6: 197, 7: 207, 8: 217, 9: 227, 10: 237, 11: 247, 12: 257, 13: 267},
    14.5: {1: 151, 2: 162, 3: 172, 4: 183, 5: 193, 6: 204, 7: 214, 8: 225, 9: 235, 10: 246, 11: 256, 12: 268, 13: 280},
    15: {1: 155, 2: 166, 3: 177, 4: 188, 5: 199, 6: 210, 7: 221, 8: 232, 9: 243, 10: 254, 11: 265, 12: 279, 13: 292},
    15.5: {1: 162, 2: 174, 3: 185, 4: 197, 5: 208, 6: 220, 7: 231, 8: 243, 9: 254, 10: 267, 11: 279, 12: 293, 13: 307},
    16: {1: 168, 2: 180, 3: 192, 4: 204, 5: 216, 6: 228, 7: 240, 8: 252, 9: 264, 10: 279, 11: 293, 12: 308, 13: 322},
    16.5: {1: 172, 2: 185, 3: 197, 4: 210, 5: 222, 6: 235, 7: 247, 8: 261, 9: 275, 10: 290, 11: 305, 12: 320, 13: 335},
    17: {1: 176, 2: 189, 3: 202, 4: 215, 5: 228, 6: 241, 7: 254, 8: 270, 9: 285, 10: 301, 11: 316, 12: 332, 13: 347},
}

# Communication/Frame points table
COMMUNICATION_FRAME_TABLE = {
    1: {1: 10, 2: 18, 3: 25, 4: 30},
    2: {1: 22, 2: 33, 3: 35, 4: 45},
    3: {1: 33, 2: 40, 3: 50, 4: 55},
    4: {1: 45, 2: 55, 3: 65, 4: 75},
    5: {1: 60, 2: 70, 3: 80, 4: 95},
}

# Innovation/Complexity points table
INNOVATION_COMPLEXITY_TABLE = {
    1: {1: 10, 2: 13, 3: 18, 4: 25},
    2: {1: 18, 2: 23, 3: 25, 4: 30},
    3: {1: 33, 2: 38, 3: 40, 4: 45},
    4: {1: 45, 2: 53, 3: 55, 4: 65},
    5: {1: 60, 2: 75, 3: 80, 4: 95},
    6: {1: 80, 2: 95, 3: 110, 4: 140},
}

# Knowledge/Teams points table
KNOWLEDGE_TEAMS_TABLE = {
    1: {1: 15, 1.5: 23, 2: 30, 2.5: 45, 3: 60},
    2: {1: 30, 1.5: 40, 2: 48, 2.5: 63, 3: 78},
    3: {1: 50, 1.5: 58, 2: 65, 2.5: 80, 3: 95},
    4: {1: 63, 1.5: 70, 2: 78, 2.5: 93, 3: 108},
    5: {1: 75, 1.5: 83, 2: 90, 2.5: 105, 3: 120},
    6: {1: 90, 1.5: 113, 2: 125, 2.5: 138, 3: 150},
    7: {1: 113, 1.5: 131, 2: 148, 2.5: 161, 3: 173},
    8: {1: 180, 1.5: 198, 2: 215, 2.5: 228, 3: 240},
}

# Breadth points
BREADTH_TABLE = {1: 0, 1.5: 5, 2: 10, 2.5: 15, 3: 20}

# IPE score to level mapping
IPE_LEVEL_MAPPING = [
    (40, 41, 1), (42, 43, 2), (44, 45, 3), (46, 47, 4),
    (48, 50, 5), (51, 52, 6), (53, 55, 7), (56, 57, 8),
    (58, 59, 9), (60, 61, 10), (62, 65, 11), (66, 73, 12)
]

###############################
# Calibration Examples       #
###############################

CALIBRATION_EXAMPLES = """
These are calibrated examples from your organization to help guide ratings:

EXAMPLE 1: Entry-Level Software Engineer (Level 3)
- Role: Implement basic software modules, learns established processes
- Impact: 1 (Delivery - executes code under guidance)
- Contribution: 2 (Some - code contributions but indirect impact)
- Communication: 1 (Convey - internal, shared interests)
- Frame: 1 (Internal shared interests)
- Innovation: 1 (Follow - follows established procedures)
- Complexity: 1 (Defined - works within module scope)
- Knowledge: 3 (Broad - has bachelor's degree, 0 years experience)
- Teams: 1 (Team member, no leadership)
- Expected IPE: 40-41, Level 1-2

EXAMPLE 2: Sr. Accounting Controller (Level 6)
- Role: Leads accounting function, designs processes, leads projects, approx 7-10 years experience
- Impact: 3 (Tactical - process/standard improvements within accounting function, 16-20% impact)
- Contribution: 4 (Significant - marked contribution to defining new accounting procedures)
- Communication: 3 (Influence - persuades on accounting practices internally, some skepticism)
- Frame: 1 (Internal shared interests - finance team)
- Innovation: 3 (Modify - updates/modifies accounting methods, improves process design)
- Complexity: 2 (Difficult - coordination across operational and financial dimensions within accounting)
- Knowledge: 5 (Professional Standard - mastery of accounting discipline + org practices, 5-8 years)
- Teams: 1 (Individual contributor / project lead on accounting matters)
- Expected IPE: 50-51, Level 5-6

EXAMPLE 3: Division Controller / Finance Head (Level 8)
- Role: Leads entire division accounting and finance, sets policy, manages multiple team members
- Impact: 4 (Strategic - influences division strategy and results, 15-20% impact at BU level)
- Contribution: 4 (Significant - marked contribution to BU financial strategy)
- Communication: 4 (Negotiate - negotiates with division stakeholders, discusses trade-offs)
- Frame: 1 or 3 (Internal, may be shared or divergent depending on division dynamics)
- Innovation: 4 (Improve - significantly improves financial processes and systems at division level)
- Complexity: 3 (Complex - addresses operational, financial, and organizational dimensions)
- Knowledge: 6 (Org. Generalist - broad financial expertise across division, 8-12 years)
- Teams: 3 (Teams manager - leads multiple accounting teams)
- Expected IPE: 56-57, Level 8
"""

###############################
# Claude API Helper          #
###############################

def query_claude_json(prompt: str, system_prompt: str = "", temperature: float = 0.3) -> dict:
    """Query Claude with JSON response format."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY environment variable.")
    
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 2000,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
    }
    
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=60
    )
    
    if resp.status_code >= 400:
        raise RuntimeError(f"Claude API error {resp.status_code}: {resp.text}")
    
    data = resp.json()
    text = data["content"][0]["text"]
    
    # Extract JSON from response
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    
    raise RuntimeError(f"Could not parse JSON from Claude response: {text}")

def query_claude_text(prompt: str, system_prompt: str = "", temperature: float = 0.3) -> str:
    """Query Claude with text response."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY environment variable.")
    
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 3000,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
    }
    
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=60
    )
    
    if resp.status_code >= 400:
        raise RuntimeError(f"Claude API error {resp.status_code}: {resp.text}")
    
    data = resp.json()
    return data["content"][0]["text"].strip()

###############################
# Evaluation Functions       #
###############################

def evaluate_dimensions(
    title: str, purpose: str, deliverables: str, decision_auth: str,
    people: float, financial: str, stakeholders: str, background: str,
    org_context: str = ""
) -> Tuple[Dict[str, float], Dict[str, str], List[str]]:
    """Evaluate IPE dimensions using Claude with official Mercer definitions."""
    
    system_prompt = f"""You are an expert in Mercer IPE job evaluation. Use ONLY the official Mercer IPE definitions provided.

{IMPACT_DEFINITIONS}

{CONTRIBUTION_DEFINITIONS}

{COMMUNICATION_DEFINITIONS}

{FRAME_DEFINITIONS}

{INNOVATION_DEFINITIONS}

{COMPLEXITY_DEFINITIONS}

{KNOWLEDGE_DEFINITIONS}

{TEAMS_DEFINITIONS}

{BREADTH_DEFINITIONS}

{CALIBRATION_EXAMPLES}

CRITICAL GUARDRAILS:
1. Impact 4-5 requires explicit DIVISION or CORPORATE-level scope signals. Functional leaders = Impact 2-3 max.
2. Entry-level white-collar with bachelor's degree = Knowledge 3-4 minimum, NOT Knowledge 1-2.
3. ICs without team leadership = Teams 1. Project leads = Teams 1.5.
4. Contribution rated in CONTEXT of Impact level (not in isolation).
5. Within-function work = Innovation 1-3 maximum. Cross-functional or end-to-end = Innovation 3-4+.
6. Complexity increases with DIMENSIONAL breadth (single area=1, two dimensions=3, all three=4).

DIMENSION VALUE CONSTRAINTS:
- Impact: INTEGER ONLY (1, 2, 3, 4, or 5) - NO HALF STEPS
- All other dimensions: Half steps allowed (e.g., 1.5, 2.5, 3.5)
- Breadth: 1, 1.5, 2, 2.5, 3
- Teams: 1, 1.5, 2, 2.5, 3

Return ONLY valid JSON with no additional text."""
    
    prompt = f"""
Evaluate this job description against Mercer IPE framework.

JOB INFORMATION:
Title: {title}
Purpose: {purpose}
Deliverables: {deliverables}
Decision Authority: {decision_auth}
People Responsibility: {people}
Financial Responsibility: {financial}
Stakeholders: {stakeholders}
Background: {background}
{f"Organizational Context: {org_context}" if org_context else ""}

Return JSON with exactly this structure:
{{
  "impact": {{"value": X, "reasoning": "..."}},
  "contribution": {{"value": X, "reasoning": "..."}},
  "communication": {{"value": X, "reasoning": "..."}},
  "frame": {{"value": X, "reasoning": "..."}},
  "innovation": {{"value": X, "reasoning": "..."}},
  "complexity": {{"value": X, "reasoning": "..."}},
  "knowledge": {{"value": X, "reasoning": "..."}},
  "teams": {{"value": X, "reasoning": "..."}},
  "breadth": {{"value": X, "reasoning": "..."}},
  "guardrail_notes": ["note1", "note2"] or []
}}

All values must be valid numbers matching IPE dimension ranges.
Breadth: 1, 1.5, 2, 2.5, 3
Teams: 1, 1.5, 2, 2.5, 3
"""
    
    result = query_claude_json(prompt, system_prompt, temperature=0.2)
    
    ratings = {}
    justifications = {}
    guardrail_notes = result.get("guardrail_notes", [])
    
    for dim in ["impact", "contribution", "communication", "frame", "innovation", "complexity", "knowledge", "teams", "breadth"]:
        if dim in result:
            ratings[dim] = result[dim].get("value", 0)
            justifications[dim] = result[dim].get("reasoning", "No reasoning provided")
    
    return ratings, justifications, guardrail_notes

def calculate_ipe_score(ratings: Dict[str, float], size: float) -> Tuple[int, Dict[str, float]]:
    """Calculate IPE score from ratings."""
    
    # Step 1: Get intermediate value from Impact × Contribution
    impact = int(ratings.get("impact", 1))
    contribution = ratings.get("contribution", 1)
    
    intermediate_value = IMPACT_CONTRIBUTION_TABLE.get(impact, {}).get(contribution, 1)
    
    # Step 2: Get final Impact/Contribution/Size points using intermediate value × Size
    impact_contrib_size_pts = IMPACT_SIZE_TABLE.get(intermediate_value, {}).get(int(size), 0)
    
    # Get other dimension points
    comm_frame_pts = COMMUNICATION_FRAME_TABLE.get(
        int(ratings.get("communication", 1)), {}
    ).get(ratings.get("frame", 1), 0)
    
    innov_complex_pts = INNOVATION_COMPLEXITY_TABLE.get(
        int(ratings.get("innovation", 1)), {}
    ).get(ratings.get("complexity", 1), 0)
    
    knowledge_teams_pts = KNOWLEDGE_TEAMS_TABLE.get(
        int(ratings.get("knowledge", 1)), {}
    ).get(ratings.get("teams", 1), 0)
    
    breadth_pts = BREADTH_TABLE.get(ratings.get("breadth", 1), 0)
    
    # Total points (Size is already included in impact_contrib_size_pts)
    total_pts = (impact_contrib_size_pts + comm_frame_pts + innov_complex_pts + 
                 knowledge_teams_pts + breadth_pts)
    
    # Calculate IPE score
    if total_pts > 26:
        ipe_score = int((total_pts - 26) / 25 + 40)
    else:
        ipe_score = None
    
    return ipe_score, {
        "impact_contribution_size": impact_contrib_size_pts,
        "intermediate_value": intermediate_value,
        "communication_frame": comm_frame_pts,
        "innovation_complexity": innov_complex_pts,
        "knowledge_teams": knowledge_teams_pts,
        "breadth": breadth_pts,
        "size": size,
        "total": total_pts,
    }

def score_to_level(ipe_score: int) -> int:
    """Convert IPE score to level 1-12."""
    for lo, hi, level in IPE_LEVEL_MAPPING:
        if lo <= ipe_score <= hi:
            return level
    return 1

###############################
# Display Helper             #
###############################

def display_evaluation_results(ipe_score: int, breakdown: Dict, ratings: Dict, 
                               justifications: Dict, guardrails: List[str], title: str):
    """Display evaluation results in a consistent format."""
    level = score_to_level(ipe_score)
    
    # Display results
    st.success("✅ Evaluation Complete")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Points", f"{breakdown['total']:.0f}")
    with col2:
        st.metric("IPE Score", ipe_score)
    with col3:
        st.metric("Job Level", f"Level {level}")
    
    # Guardrail notes (if any)
    if guardrails:
        st.info("**Guardrail Notes:**\n" + "\n".join(f"- {note}" for note in guardrails))
    
    # Point breakdown
    with st.expander("📊 Point Breakdown", expanded=True):
        breakdown_df = pd.DataFrame([
            {"Dimension": "Impact × Contribution × Size", "Points": breakdown["impact_contribution_size"], "Detail": f"(intermediate: {breakdown.get('intermediate_value', 'N/A')}, size: {breakdown['size']})"},
            {"Dimension": "Communication × Frame", "Points": breakdown["communication_frame"], "Detail": ""},
            {"Dimension": "Innovation × Complexity", "Points": breakdown["innovation_complexity"], "Detail": ""},
            {"Dimension": "Knowledge × Teams × Breadth", "Points": breakdown["knowledge_teams"] + breakdown["breadth"], "Detail": f"(knowledge×teams: {breakdown['knowledge_teams']}, breadth: {breakdown['breadth']})"},
            {"Dimension": "**TOTAL**", "Points": breakdown["total"], "Detail": ""},
        ])
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    
    # Dimension ratings with reasoning
    st.markdown("### 📋 Dimension Ratings & Reasoning")
    
    # Impact & Contribution
    with st.expander("🎯 Impact & Contribution", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Impact:** {ratings.get('impact', 'N/A')}")
            st.caption(justifications.get('impact', 'No reasoning provided'))
        with col2:
            st.markdown(f"**Contribution:** {ratings.get('contribution', 'N/A')}")
            st.caption(justifications.get('contribution', 'No reasoning provided'))
    
    # Communication & Frame
    with st.expander("💬 Communication & Frame"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Communication:** {ratings.get('communication', 'N/A')}")
            st.caption(justifications.get('communication', 'No reasoning provided'))
        with col2:
            st.markdown(f"**Frame:** {ratings.get('frame', 'N/A')}")
            st.caption(justifications.get('frame', 'No reasoning provided'))
    
    # Innovation & Complexity
    with st.expander("💡 Innovation & Complexity"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Innovation:** {ratings.get('innovation', 'N/A')}")
            st.caption(justifications.get('innovation', 'No reasoning provided'))
        with col2:
            st.markdown(f"**Complexity:** {ratings.get('complexity', 'N/A')}")
            st.caption(justifications.get('complexity', 'No reasoning provided'))
    
    # Knowledge, Teams, Breadth
    with st.expander("🎓 Knowledge, Teams & Breadth"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Knowledge:** {ratings.get('knowledge', 'N/A')}")
            st.caption(justifications.get('knowledge', 'No reasoning provided'))
        with col2:
            st.markdown(f"**Teams:** {ratings.get('teams', 'N/A')}")
            st.caption(justifications.get('teams', 'No reasoning provided'))
        with col3:
            st.markdown(f"**Breadth:** {ratings.get('breadth', 'N/A')}")
            st.caption(justifications.get('breadth', 'No reasoning provided'))
    
    # Export results
    st.markdown("---")
    results_json = {
        "title": title,
        "ipe_score": ipe_score,
        "level": level,
        "total_points": breakdown["total"],
        "ratings": ratings,
        "justifications": justifications,
        "guardrails": guardrails,
        "breakdown": breakdown
    }
    
    st.download_button(
        "📥 Download Evaluation Results (JSON)",
        data=json.dumps(results_json, indent=2),
        file_name=f"ipe_evaluation_{title.replace(' ', '_')}.json",
        mime="application/json"
    )

###############################
# Streamlit UI               #
###############################

def main():
    st.title("📋 IPE Job Description Evaluator (Official Mercer Framework)")
    st.caption("Using official Mercer IPE definitions with organizational context")
    
    with st.expander("📚 About This Tool", expanded=False):
        st.markdown("""
This tool evaluates job descriptions using the **official Mercer IPE framework** with proper guardrails.

**Key Improvements:**
- Uses official Mercer definitions (not simplified versions)
- Proper guardrails for all dimensions
- Supports fractional team responsibility (1, 1.5, 2, 2.5, 3)
- Detailed reasoning for each rating
- Calibrated against your organization's examples

**How to Use:**
1. Choose evaluation mode
2. Enter job details or paste existing JD
3. Click "Evaluate"
4. Review the IPE score, level, and reasoning for each dimension
        """)
    
    # Mode selection
    mode = st.radio("Select Mode:", ["Evaluate from Structured Inputs", "Evaluate Existing JD"], horizontal=True)
    
    if mode == "Evaluate Existing JD":
        # Simple mode: paste full JD
        st.header("Paste Job Description")
        jd_text = st.text_area("Job Description (paste the complete JD here) *", height=400)
        job_title = st.text_input("Job Title (optional, helps with context)", "")
        
        st.markdown("### Organizational Context")
        col1, col2, col3 = st.columns(3)
        with col1:
            size = st.slider("Organization Size Score (1-13)", min_value=1.0, max_value=13.0, value=7.0, step=0.5)
        with col2:
            teams_input = st.selectbox("Team Responsibility", [1, 1.5, 2, 2.5, 3], index=0, format_func=lambda x: {
                1: "1 - Individual Contributor",
                1.5: "1.5 - Project Lead",
                2: "2 - Team Leader",
                2.5: "2.5 - Hybrid Manager",
                3: "3 - Teams Manager"
            }.get(x, str(x)))
        with col3:
            breadth_input = st.selectbox("Breadth", [1, 1.5, 2, 2.5, 3], index=0, format_func=lambda x: {
                1: "1 - Domestic",
                1.5: "1.5 - Sub-Region",
                2: "2 - Regional",
                2.5: "2.5 - Multiple Regions",
                3: "3 - Global"
            }.get(x, str(x)))
        
        if st.button("🔍 Evaluate IPE", key="eval_jd_btn", use_container_width=True):
            if not jd_text.strip():
                st.error("Please paste a job description to evaluate.")
            else:
                with st.spinner("Evaluating using official Mercer IPE framework..."):
                    try:
                        # Extract basic info from JD for context
                        system_prompt = f"""You are an expert in Mercer IPE job evaluation. Use ONLY the official Mercer IPE definitions provided.

{IMPACT_DEFINITIONS}

{CONTRIBUTION_DEFINITIONS}

{COMMUNICATION_DEFINITIONS}

{FRAME_DEFINITIONS}

{INNOVATION_DEFINITIONS}

{COMPLEXITY_DEFINITIONS}

{KNOWLEDGE_DEFINITIONS}

{TEAMS_DEFINITIONS}

{BREADTH_DEFINITIONS}

{CALIBRATION_EXAMPLES}

CRITICAL GUARDRAILS:
1. Impact 4-5 requires explicit DIVISION or CORPORATE-level scope signals. Functional leaders = Impact 2-3 max.
2. Entry-level white-collar with bachelor's degree = Knowledge 3-4 minimum, NOT Knowledge 1-2.
3. ICs without team leadership = Teams 1. Project leads = Teams 1.5.
4. Contribution rated in CONTEXT of Impact level (not in isolation).
5. Within-function work = Innovation 1-3 maximum. Cross-functional or end-to-end = Innovation 3-4+.
6. Complexity increases with DIMENSIONAL breadth (single area=1, two dimensions=3, all three=4).

DIMENSION VALUE CONSTRAINTS:
- Impact: INTEGER ONLY (1, 2, 3, 4, or 5) - NO HALF STEPS
- All other dimensions: Half steps allowed (e.g., 1.5, 2.5, 3.5)
- Breadth: 1, 1.5, 2, 2.5, 3
- Teams: 1, 1.5, 2, 2.5, 3

Return ONLY valid JSON with no additional text."""
                        
                        prompt = f"""
Evaluate this complete job description against Mercer IPE framework.

{f"Job Title: {job_title}" if job_title else ""}

JOB DESCRIPTION:
{jd_text}

USER PROVIDED CONTEXT:
- Teams responsibility: {teams_input}
- Breadth: {breadth_input}

Return JSON with exactly this structure:
{{
  "impact": {{"value": X, "reasoning": "..."}},
  "contribution": {{"value": X, "reasoning": "..."}},
  "communication": {{"value": X, "reasoning": "..."}},
  "frame": {{"value": X, "reasoning": "..."}},
  "innovation": {{"value": X, "reasoning": "..."}},
  "complexity": {{"value": X, "reasoning": "..."}},
  "knowledge": {{"value": X, "reasoning": "..."}},
  "guardrail_notes": ["note1", "note2"] or []
}}

Note: Teams and Breadth are provided by user, so don't evaluate those. Only evaluate the 7 dimensions listed above.
"""
                        
                        result = query_claude_json(prompt, system_prompt, temperature=0.2)
                        
                        ratings = {}
                        justifications = {}
                        guardrails = result.get("guardrail_notes", [])
                        
                        # Get ratings from Claude
                        for dim in ["impact", "contribution", "communication", "frame", "innovation", "complexity", "knowledge"]:
                            if dim in result:
                                ratings[dim] = result[dim].get("value", 0)
                                justifications[dim] = result[dim].get("reasoning", "No reasoning provided")
                        
                        # Add user-provided values
                        ratings["teams"] = teams_input
                        ratings["breadth"] = breadth_input
                        justifications["teams"] = f"User-provided: {teams_input}"
                        justifications["breadth"] = f"User-provided: {breadth_input}"
                        
                        # Calculate IPE score
                        ipe_score, breakdown = calculate_ipe_score(ratings, size)
                        
                        if ipe_score:
                            display_evaluation_results(ipe_score, breakdown, ratings, justifications, guardrails, job_title or "Job")
                        else:
                            st.error("Could not calculate valid IPE score. Check your inputs.")
                            
                    except Exception as e:
                        st.error(f"Error during evaluation: {str(e)}")
                        st.exception(e)
    
    else:
        # Original structured input mode
        evaluate_from_structured_inputs()

def evaluate_from_structured_inputs():
    """Original mode with structured input fields."""
    # Input form
    st.header("Job Information")
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input("Job Title *")
        purpose = st.text_area("Purpose of Role (3-6 sentences) *", height=80)
        deliverables = st.text_area("Top Deliverables (one per line) *", height=80)
        
    with col2:
        decision = st.text_area("Decision-Making Authority *", height=80)
        people = st.selectbox("People Responsibility *", [1, 1.5, 2, 2.5, 3], format_func=lambda x: {
            1: "1 - Individual Contributor",
            1.5: "1.5 - Project Lead (no direct reports)",
            2: "2 - Team Leader (3-8 people)",
            2.5: "2.5 - Hybrid Manager",
            3: "3 - Teams Manager (multiple teams)"
        }.get(x, str(x)))
        financial = st.text_input("Financial Responsibility (e.g., $5M budget, P&L ownership)")
        stakeholders = st.text_area("Main Stakeholders", height=100)
    
    background = st.text_area("Required Background / Experience *", height=100)
    org_context = st.text_area("Organizational Context (optional)", height=100, placeholder="Any additional context about the role or organization")
    
    # Size input
    st.markdown("### Organizational Context")
    size = st.slider("Organization Size Score (1-13)", min_value=1.0, max_value=13.0, value=7.0, step=0.5)
    
    # Evaluate button
    if st.button("🔍 Evaluate IPE", key="eval_btn", use_container_width=True):
        # Validation
        missing = []
        if not title: missing.append("Job Title")
        if not purpose: missing.append("Purpose")
        if not deliverables: missing.append("Deliverables")
        if not decision: missing.append("Decision-Making Authority")
        if not background: missing.append("Background")
        
        if missing:
            st.error(f"Please fill in required fields: {', '.join(missing)}")
        else:
            with st.spinner("Evaluating using official Mercer IPE framework..."):
                try:
                    # Evaluate dimensions
                    ratings, justifications, guardrails = evaluate_dimensions(
                        title, purpose, deliverables, decision,
                        people, financial, stakeholders, background,
                        org_context
                    )
                    
                    # Calculate IPE score
                    ipe_score, breakdown = calculate_ipe_score(ratings, size)
                    
                    if ipe_score:
                        level = score_to_level(ipe_score)
                        display_evaluation_results(ipe_score, breakdown, ratings, justifications, guardrails, title)
                    else:
                        st.error("Could not calculate valid IPE score. Check your inputs.")
                        
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    main()
