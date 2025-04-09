const BASE_URL_CUSTOMER = 'http://localhost:8000';

interface LoginResponse {
  token: string;
  user: {
    id: string;
    name: string;
    phone: string;
  };
}

export const authService = {
  async sendOTP(mobile: string): Promise<{ success: boolean; message: string }> {
    try {
      const response = await fetch(`${BASE_URL_CUSTOMER}/auth/sendotp`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ mobile }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Failed to send OTP');
      }

      return { success: true, message: 'OTP sent successfully' };
    } catch (error: any) {
      throw new Error(error.message || 'Failed to send OTP');
    }
  },

  async verifyOTP(mobile: string, otp: string): Promise<LoginResponse> {
    try {
      console.log('Sending OTP verification request:', { mobile, otp, deviceId: 'postman' });

      const response = await fetch(`${BASE_URL_CUSTOMER}/auth/verifyotp`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mobile,
          otp: parseInt(otp, 10), // Ensure OTP is sent as a number
          deviceId: 'postman', // Ensure deviceId is correct
        }),
      });

      console.log('Response status:', response.status);
      const data = await response.json();
      console.log('Response data:', data);
      console.log(response,"getData")

      if (!response.ok) {
        console.error('Server error:', data.message || 'Unknown error');
        throw new Error(data.message || 'Failed to verify OTP');
      }

      // if (!data.token) {
      //   console.error('Missing token in response:', data);
      //   throw new Error('Invalid response from server: Missing token');
      // }

      localStorage.setItem('auth_token', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));

      return data;
    } catch (error: any) {
      console.error('Error during OTP verification:', error.message);
      throw new Error(error.message || 'Failed to verify OTP');
    }
  },

  async resendOTP(mobile: string): Promise<{ success: boolean; message: string }> {
    try {
      const response = await fetch(`${BASE_URL_CUSTOMER}/auth/resendotp`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ mobile }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Failed to resend OTP');
      }

      return { success: true, message: 'OTP resent successfully' };
    } catch (error: any) {
      throw new Error(error.message || 'Failed to resend OTP');
    }
  },

  async logout(): Promise<void> {
    try {
      const token = this.getToken();
      if (!token) return;

      const response = await fetch(`${BASE_URL_CUSTOMER}/auth/logout`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to logout');
      }

      this.clearAuth();
    } catch (error) {
      console.error('Error in logout:', error);
      this.clearAuth(); // Clear auth even if API call fails
    }
  },

  async refreshToken(): Promise<string> {
    try {
      const token = this.getToken();
      if (!token) throw new Error('No token found');

      const response = await fetch(`${BASE_URL_CUSTOMER}/auth/refresh-token`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to refresh token');
      }

      const data = await response.json();
      localStorage.setItem('auth_token', data.token);
      return data.token;
    } catch (error) {
      console.error('Error in refreshToken:', error);
      this.clearAuth();
      throw error;
    }
  },

  async getProfile(): Promise<any> {
    try {
      const token = this.getToken();
      if (!token) throw new Error('No token found');

      const response = await fetch(`${BASE_URL_CUSTOMER}/auth/profile`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to get profile');
      }

      return await response.json();
    } catch (error) {
      console.error('Error in getProfile:', error);
      throw error;
    }
  },

  clearAuth() {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
  },

  getToken(): string | null {
    return localStorage.getItem('auth_token');
  },

  getUser() {
    const userStr = localStorage.getItem('user');
    return userStr ? JSON.parse(userStr) : null;
  }
};

export const verifyOTP = async (mobileNumber: string, otp: number) => {
    try {
        console.log('Verifying OTP:', { mobileNumber, otp });
        const response = await fetch('https://api.Ṧ.ai/api/v1/auth/verify-otp', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                mobileNumber,
                otp,
                deviceId: "web"
            }),
        });

        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Raw response data:', data);

        if (!response.ok) {
            throw new Error(data.message || 'Failed to verify OTP');
        }

        if (!data.data?.token) {
            throw new Error('Invalid response from server: Missing token');
        }

        // Store token and user data
        localStorage.setItem('token', data.data.token);
        localStorage.setItem('user', JSON.stringify(data.data.user || {}));

        return {
            token: data.data.token,
            user: data.data.user || {}
        };
    } catch (error) {
        console.error('Error in verifyOTP:', error);
        throw error;
    }
};