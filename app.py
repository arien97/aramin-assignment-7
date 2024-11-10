from flask import Flask, render_template, request, url_for, session
from flask_session import Session
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import os

secret_key = os.urandom(24)  # Generates a 24-byte string
print("secret key ", secret_key)

app = Flask(__name__)
app.secret_key = secret_key  # Replace with your own secret key, needed for session management

# Configure server-side session storage
app.config["SESSION_TYPE"] = "filesystem"  # You can also use "redis", "memcached", etc.
app.config["SESSION_FILE_DIR"] = "./flask_session"  # Directory to store session files
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True

Session(app)

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # TODO 1: Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)  # Generates random values between 0 and 1

    # TODO 2: Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    # Y = beta0 + beta1 * X + mu + error term
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)  # Adds normal error with mean 0 and variance sigma2

    # TODO 3: Fit a linear regression model to X and Y
    model = LinearRegression()  # Initialize the LinearRegression model
    model.fit(X.reshape(-1, 1), Y)  # Fit the model to X and Y
    slope = model.coef_[0]  # Extract the slope (coefficient) from the fitted model
    intercept = model.intercept_  # Extract the intercept from the fitted model

    # TODO 4: Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    # Replace with code to generate and save the scatter plot
    plt.scatter(X, Y, color="blue", label="Data points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label=f"Regression line: Y = {slope:.2f}X + {intercept:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Regression Line")
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()

    # TODO 5: Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # TODO 6: Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.rand(N) # Replace with code to generate simulated X values
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)  # Generate Y_sim values  # Replace with code to generate simulated Y values

        # TODO 7: Fit linear regression to simulated data and store slope and intercept
        sim_model = LinearRegression()  # Replace with code to fit the model
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)  # Extract slope from sim_model
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_  # Extract intercept from sim_model

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # TODO 8: Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"
    # Replace with code to generate and save the histogram plot
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", label=f"Observed Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", label=f"Observed Intercept: {intercept:.2f}")
    plt.legend()
    plt.savefig(plot2_path)
    plt.close()

    # TODO 9: Return data needed for further analysis, including slopes and intercepts
    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = sum(abs(s) > abs(slope) for s in slopes) / S # Replace with code to calculate proportion of slopes more extreme than observed
    intercept_extreme = sum(abs(i) > abs(intercept) for i in intercepts) / S  # Replace with code to calculate proportion of intercepts more extreme than observed

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)
        
        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Debugging: Print session data
        print("Session data set:", session)

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # # Retrieve data from session
    # N = int(session.get("N"))
    # print(N)
    # S = int(session.get("S"))
    # slope = float(session.get("slope"))
    # intercept = float(session.get("intercept"))
    # slopes = session.get("slopes")
    # intercepts = session.get("intercepts")
    # beta0 = float(session.get("beta0"))
    # beta1 = float(session.get("beta1"))

    # Retrieve data from session
    N = session.get("N")
    S = session.get("S")
    slope = session.get("slope")
    intercept = session.get("intercept")
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = session.get("beta0")
    beta1 = session.get("beta1")

    # Debugging: Print session data
    print("Session data retrieved:", session)

    # Check if any required session data is missing
    if any(x is None for x in [N, S, slope, intercept, slopes, intercepts, beta0, beta1]):
        print("Missing session data")
        return "Session data missing. Please ensure data generation is completed before hypothesis testing.", 400
    
    # Debugging: Print individual session variables
    print(f"N: {N}, S: {S}, slope: {slope}, intercept: {intercept}, beta0: {beta0}, beta1: {beta1}")
    
    # # Convert session data to required types
    # N = int(N)
    # S = int(S)
    # slope = float(slope)
    # intercept = float(intercept)
    # slopes = list(map(float, slopes))
    # intercepts = list(map(float, intercepts))
    # beta0 = float(beta0)
    # beta1 = float(beta1)
     # Convert session data to required types
    try:
        N = int(N)
        S = int(S)
        slope = float(slope)
        intercept = float(intercept)
        slopes = list(map(float, slopes))
        intercepts = list(map(float, intercepts))
        beta0 = float(beta0)
        beta1 = float(beta1)
    except (TypeError, ValueError) as e:
        print(f"Error converting session data: {e}")
        return "Error converting session data.", 400

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # TODO 10: Calculate p-value based on test type
    if test_type == "!=":
        p_value = sum(abs(stat) >= abs(observed_stat) for stat in simulated_stats) / S
    elif test_type == ">":
        p_value = sum(stat >= observed_stat for stat in simulated_stats) / S
    elif test_type == "<":
        p_value = sum(stat <= observed_stat for stat in simulated_stats) / S
    else:
        p_value = None

    # TODO 11: If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    fun_message = "This seems very unlikely..." if p_value <= 0.0001 else ""

    # TODO 12: Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    # Replace with code to generate and save the plot
    plt.hist(simulated_stats, bins=20, color="gray", alpha=0.6, label="Simulated Statistics")
    plt.axvline(observed_stat, color="red", linestyle="--", label="Observed " + parameter + ": " + str(observed_stat))
    plt.axvline(hypothesized_value, color="blue", linestyle="-", label="Hypothesized value: " + str(hypothesized_value))
    plt.title("Distribution of Simulated Statistics")
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()


    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        # TODO 13: Uncomment the following lines when implemented
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Convert confidence level from percentage to proportion
    confidence_level /= 100

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # TODO 14: Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)

    # TODO 15: Calculate confidence interval for the parameter estimate
    # Use the t-distribution and confidence_level
    ci_lower, ci_upper = stats.t.interval(confidence_level, df=S-1, loc=mean_estimate, scale=std_estimate / np.sqrt(S))

    # TODO 16: Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # TODO 17: Plot the individual estimates as gray points and confidence interval
    # Plot the mean estimate as a colored point which changes if the true parameter is included
    # Plot the confidence interval as a horizontal line
    # Plot the true parameter value
    plot4_path = "static/plot4.png"
    # Write code here to generate and save the plot
    plt.scatter(estimates, np.zeros_like(estimates), color="gray", alpha=0.5, label="Estimates")
    
    # mean_color = "blue" if includes_true else "red"
    plt.scatter([mean_estimate], [0], color="blue", s=100, zorder=5, label="Mean Estimate")

    # Plot the confidence interval as a horizontal line
    plt.hlines(0, ci_lower, ci_upper, color="blue", linestyle="-", linewidth=5, label="Confidence Interval")
    # plt.axvline(mean_estimate, color="green", linestyle="-", label="Mean Estimate")
   
    # plt.axvline(ci_lower, color="blue", linestyle="--", label="CI Lower Bound")
    # plt.axvline(ci_upper, color="blue", linestyle="--", label="CI Upper Bound")
    plt.axvline(true_param, color="pink", linestyle=":", label="True " + parameter)
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()


    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level * 100,  # Convert back to percentage for display
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)
