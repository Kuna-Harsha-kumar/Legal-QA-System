The current model is using CAUD database.The model can be trained only on a single document.The below are the sample results : 


The below results shows the results when there are labels in the document,so that supervised method is followed

<xgboost.core.Booster object at 0x788777f99a00>

🟢 DETECTED MODE: SUPERVISED CLASSIFICATION
📌 Evaluating using true labels (no train/test split)

📊 SUPERVISED METRICS:
Accuracy :  0.5510
Precision: 0.4123
Recall   : 0.5510
F1 Score : 0.4510

The below results show  the results when there are no labels and model is following unsupervised method(regression)

<xgboost.core.Booster object at 0x7ac5fdaa0bf0>

🟡 DETECTED MODE: UNSUPERVISED REGRESSION
📌 Evaluating using synthetic labels

📊 SYNTHETIC REGRESSION METRICS:
RMSE: 0.000291
MAE : 0.000222
R²  : 0.843736

The below are the retreival output rankings procuded.The model is not generating the text but retriving the top tanked legal clauses,


❓ QUESTION: Is there any limitation of liability?
🔹 Top 1 (score=0.1895)
(a) Carrier shall, at its sole cost and expense, procure and maintain liability insurance with a reputable and financially responsible insurance carrier or carriers properly insuring Carrier against liability and claims for injuries to persons (including injuries resulting in death) and for damage to property in amounts not less than the Minimum Levels of Financial Responsibility for Motor Carriers prescribed by the U. S. Department of Transportation (49 CFR (S)387 et seq.
------------------------------------------------------------
🔹 Top 2 (score=0.1667)
The term force majeure shall include, without limitation, acts of God and the public enemy, the elements, fire, accidents, breakdown, strikes, and any other industrial, civil or public disturbance, inability to obtain materials, supplies.
------------------------------------------------------------
🔹 Top 3 (score=0.1302)
A claim must be filed with Carrier within thirty (30) days from the date the shipment in question was delivered, and (i) contain facts sufficient to identify the shipment (or shipments) involved (ii) assert the grounds for Carrier s liability for alleged loss, damage, injury, or delay, and (iii) request payment of a specified or determinable amount of money.
------------------------------------------------------------
❓ QUESTION: What is the warranty period?
🔹 Top 1 (score=0.3347)
In this case, this time shall begin at the earliest hour of the agreed arrival period if the Carrier is early or at the time of actual hookup and beginning of unloading if the Carrier arrives later than the agreed arrival period.
------------------------------------------------------------
🔹 Top 2 (score=0.2638)
A day shall be defined as a twenty-four hour period commencing at 12:01 a.m. local time at the place the equipment is to be delivered.
------------------------------------------------------------
🔹 Top 3 (score=0.2571)
The exception to this computation of time shall be when, by mutual agreement of Carrier, Consignor and Consignee, an arrival period is accepted and not met by the Carrier.
------------------------------------------------------------
❓ QUESTION: What happens if we terminate the contract?
🔹 Top 1 (score=0.2573)
(a) if either party should make a general assignment for the benefit of creditors or if a receiver should be appointed on account of the insolvency of either party, the other party may, without prejudice to any other right or remedy, terminate this contract upon seven (7) days prior written notice.
------------------------------------------------------------
🔹 Top 2 (score=0.2316)
As compensation for the services provided by Carrier under this contract, Shipper shall pay Carrier in accordance with 1) Rate Appendices making reference to this contract which shall from time to time be agreed to between the parties and 2) Carrier s Contract Carriage Rules and Regulations attached as Exhibit A, which are incorporated in this contract by this reference for all purposes (collectively, the Schedule ).
------------------------------------------------------------
🔹 Top 3 (score=0.1625)
THIS CONTRACT IS SUBJECT TO THE TERMS AND CONDITIONS ON THE REVERSE SIDE.
------------------------------------------------------------
❓ QUESTION: Are there financial penalties or fees?
🔹 Top 1 (score=0.1508)
(a) Carrier shall, at its sole cost and expense, procure and maintain liability insurance with a reputable and financially responsible insurance carrier or carriers properly insuring Carrier against liability and claims for injuries to persons (including injuries resulting in death) and for damage to property in amounts not less than the Minimum Levels of Financial Responsibility for Motor Carriers prescribed by the U. S. Department of Transportation (49 CFR (S)387 et seq.
------------------------------------------------------------
🔹 Top 2 (score=0.1331)
The terms and conditions of this contract and all information concerning the business, customers, products, and processes of each party which may come into the possession of the other party during the course of the negotiation or performance of this contract are confidential and shall not be disclosed to any third party without the prior written consent of the other party provided, however, either party may disclose information concerning this contract to any independent public accounting firm retained to perform an annual financial audit of that party.
------------------------------------------------------------
🔹 Top 3 (score=0.0000)
The provisions of this Item will only apply on distance commodity scales in excess of 400 miles.
------------------------------------------------------------
