# Build stage
FROM node:22-alpine as build

WORKDIR /app

# Copy package files and install
COPY package.json package-lock.json ./
RUN npm install

# Copy application and build
COPY . .
RUN npm run build

# Production stage using Nginx
FROM nginx:alpine

# Copy built assets from build stage to nginx serving directory
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
